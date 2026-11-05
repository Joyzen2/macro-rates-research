"""第 19–22 步（研究报告）用的公开数据，取到 data/raw/reports/，再整理成 data/reports/。运行：uv run python -m macrob.reports_data

新浪的单合约日线大约从 2019 年起才有；不存在的合约会报错，跳过。请求之间停 1 秒。
"""
from __future__ import annotations

import shutil
import time

import pandas as pd

from .liquidity import _retry
from .probe_util import RAW

OUT = RAW / "reports"
OUT.mkdir(parents=True, exist_ok=True)
FIRST = {"IF": "2018-06", "IH": "2018-06", "IC": "2018-06", "IM": "2022-07", "T": "2018-06", "TF": "2018-06", "TS": "2018-09", "TL": "2023-06"}
QUARTERLY = {"T", "TF", "TS", "TL"}


def contracts() -> pd.DataFrame:
    import akshare as ak
    path = OUT / "futures_contracts.csv"
    have = pd.read_csv(path, dtype={"contract": str}) if path.exists() else pd.DataFrame()
    done = set(have["contract"]) if len(have) else set()
    live_from = pd.Timestamp.today() - pd.Timedelta(days=45)         # 还在交易的合约每次都重取
    rows, tried, failed = [have] if len(have) else [], 0, 0
    months = pd.period_range("2018-06", "2027-12", freq="M")
    for prod, first in FIRST.items():
        for m in months[months >= pd.Period(first)]:
            if prod in QUARTERLY and m.month % 3:
                continue
            code = f"{prod}{m.strftime('%y%m')}"
            if code in done and m.to_timestamp() < live_from:
                continue
            tried += 1
            try:
                d = ak.futures_zh_daily_sina(symbol=code)
            except Exception:
                failed += 1
                time.sleep(0.5)
                continue
            if len(d):
                rows.append(d.drop(columns=["settle"]).assign(contract=code, product=prod))
            time.sleep(1.0)
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["contract", "date"], keep="last").sort_values(["product", "contract", "date"])
    out.to_csv(path, index=False)
    print(f"合约日线 {len(out)} 行，{out['contract'].nunique()} 个合约，试了 {tried} 个，没有数据的 {failed} 个；{out['date'].min()} 至 {out['date'].max()}")
    return out


def indexes() -> None:
    import akshare as ak
    px = {}
    for name, code in (("沪深300", "sh000300"), ("上证50", "sh000016"), ("中证500", "sh000905"), ("中证1000", "sh000852")):
        d = _retry(lambda c=code: ak.stock_zh_index_daily(symbol=c))
        px[name] = pd.Series(d["close"].to_numpy(), index=pd.to_datetime(d["date"]))
        time.sleep(1)
    pd.DataFrame(px).to_csv(OUT / "index_price.csv", index_label="date")
    tr, pe = {}, {}
    end = pd.Timestamp.today().strftime("%Y%m%d")
    for name, code in (("沪深300", "H00300"), ("上证50", "H00016"), ("中证500", "H00905"), ("中证1000", "H00852")):
        d = _retry(lambda c=code: ak.stock_zh_index_hist_csindex(symbol=c, start_date="20140101", end_date=end))
        tr[name] = pd.Series(pd.to_numeric(d["收盘"]).to_numpy(), index=pd.to_datetime(d["日期"]))
        time.sleep(1)
    pd.DataFrame(tr).to_csv(OUT / "index_total_return.csv", index_label="date")
    d = _retry(lambda: ak.stock_zh_index_hist_csindex(symbol="000300", start_date="20140101", end_date=end))
    pd.DataFrame({"沪深300滚动市盈率": pd.to_numeric(d["滚动市盈率"], errors="coerce").to_numpy()}, index=pd.to_datetime(d["日期"])).to_csv(OUT / "hs300_pe.csv", index_label="date")
    print("指数", pd.DataFrame(px).index.max().date(), "全收益", pd.DataFrame(tr).index.max().date())


def others() -> None:
    import akshare as ak
    end = pd.Timestamp.today().strftime("%Y%m%d")
    _retry(lambda: ak.bond_zh_us_rate(start_date="19901219")).to_csv(OUT / "cn_us_yields.csv", index=False)
    time.sleep(1)
    fx = _retry(lambda: ak.currency_boc_safe())
    fx[["日期", "美元"]].rename(columns={"日期": "date", "美元": "每100美元兑人民币中间价"}).to_csv(OUT / "usdcny_parity.csv", index=False)
    time.sleep(1)
    _retry(lambda: ak.index_us_stock_sina(symbol=".INX"))[["date", "close"]].to_csv(OUT / "sp500.csv", index=False)
    time.sleep(1)
    gold = _retry(lambda: ak.fund_etf_hist_sina(symbol="sh518880"))
    gold[["date", "close"]].to_csv(OUT / "gold_etf_518880.csv", index=False)
    time.sleep(1)
    com = _retry(lambda: ak.stock_zh_index_hist_csindex(symbol="H11061", start_date="20140101", end_date=end))
    pd.DataFrame({"date": com["日期"], "close": com["收盘"]}).to_csv(OUT / "commodity_h11061.csv", index=False)
    time.sleep(1)
    print("中美收益率、中间价、标普、黄金、商品已存")


def snapshot(out=None) -> None:
    """把取好的数据整理成四篇报告（rpt_*）读的 data/reports/。FOMC 日期（fomc_dates.csv）不在这里生成：财经日历解析出过 7 处错误，
    现在是对过美联储官网的固定文件，新的决议手工加一行。policy_events 去掉用利率互换算出来的列。"""
    out = out or RAW.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("futures_contracts", "index_price", "index_total_return", "hs300_pe", "cn_us_yields", "usdcny_parity", "sp500", "gold_etf_518880", "commodity_h11061"):
        shutil.copy2(OUT / f"{name}.csv", out / f"{name}.csv")
    shutil.copy2(RAW / "cbond_treasury_wealth_7-10Y.csv", out / "cbond_treasury_wealth_7-10Y.csv")
    c = pd.concat([pd.read_csv(RAW / f"chinabond_curve_{y}.csv") for y in range(2015, pd.Timestamp.today().year + 1)])
    c = c[c["曲线名称"] == "中债国债收益率曲线"].drop(columns="曲线名称").rename(columns={"日期": "date"})
    c.drop_duplicates("date").sort_values("date").to_csv(out / "chinabond_treasury.csv", index=False)
    results = RAW.parent.parent / "results"
    f = pd.read_csv(results / "liquidity_daily.csv")
    f[["date", "DR001s", "DR007s", "FR001", "FR007", "SHIBOR_ON", "SHIBOR_1W", "SHIBOR_3M", "GC001", "rate7", "rate_on"]].to_csv(out / "funding_daily.csv", index=False)
    e = pd.read_csv(results / "policy_irs_events.csv", encoding="utf-8-sig")
    e[["date", "工具", "size", "direction", "分组"]].to_csv(out / "policy_events.csv", index=False)
    print(f"整理好 {len(list(out.glob('*.csv')))} 个文件 → {out}")


if __name__ == "__main__":
    for fn in (indexes, others, contracts, snapshot):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} 失败：{type(e).__name__} {str(e)[:200]}")
