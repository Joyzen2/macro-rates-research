"""第 19–22 步（研究报告）用的公开数据，取到 data/raw/reports/。运行：uv run python -m macrob.reports_data

新浪的单合约日线大约从 2019 年起才有；不存在的合约会报错，跳过。请求之间停 1 秒。
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request

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


def fomc() -> None:
    """华尔街见闻财经日历里的美联储利率决议，按月查。只留决议本身，不留讲话。"""
    rows = []
    for m in pd.period_range("2010-01", "2026-12", freq="M"):
        a, b = m.to_timestamp(), (m + 1).to_timestamp()
        url = f"https://api-one-wscn.awtmt.com/apiv1/finance/macrodatas?start={int(a.timestamp())}&end={int(b.timestamp())}"
        items = _retry(lambda: json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=25).read())["data"]["items"])
        for it in items:
            t = it.get("title", "")
            if it.get("country") == "美国" and any(k in t for k in ("利率决策", "利率决议", "利率上限", "利率下限", "宣布利率决定")) and "票委" not in t and "讲话" not in t:
                rows.append({"time": dt.datetime.fromtimestamp(it["public_date"]).strftime("%Y-%m-%d %H:%M"), "title": t, "actual": it.get("actual"), "previous": it.get("previous"), "forecast": it.get("forecast")})
        time.sleep(1.0)
    pd.DataFrame(rows).drop_duplicates().to_csv(OUT / "fomc_calendar_raw.csv", index=False)
    print("FOMC 日历", len(rows), "条")


if __name__ == "__main__":
    for fn in (indexes, others, fomc, contracts):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} 失败：{type(e).__name__} {str(e)[:200]}")
