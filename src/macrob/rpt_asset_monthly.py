"""第四篇：大类资产月报（2026 年 10 月）。以后每月一期，改 MONTH_END 即可。

uv run python -m macrob.rpt_asset_monthly
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rpt_common import load, save_csv, save_svg

REPORT = "2026-11-assets-october"
MONTH_END = pd.Timestamp("2026-10-30")
PREV_END = pd.Timestamp("2026-09-30")
YEAR_BASE = pd.Timestamp("2025-12-31")
VOL_DAYS, CORR_DAYS, ANN = 60, 60, 252


def series():
    tr = load("index_total_return")
    s = {f"{c}（全收益）": tr[c] for c in ["沪深300", "上证50", "中证500", "中证1000"]}
    s["中债 7–10 年国债财富"] = load("cbond_treasury_wealth_7-10Y")["value"]
    s["黄金 ETF"] = load("gold_etf_518880")["close"]
    s["中证商品期货综合"] = load("commodity_h11061")["close"]
    s["美元兑人民币中间价"] = load("usdcny_parity").iloc[:, 0]
    s["标普 500"] = load("sp500")["close"]
    return {k: v.dropna().sort_index() for k, v in s.items()}


def asof(s: pd.Series, d: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    x = s[s.index <= d]
    return x.index[-1], float(x.iloc[-1])


def returns_table(ser):
    rows = []
    for name, s in ser.items():
        s = s[s.index <= MONTH_END]
        d_end, v_end = asof(s, MONTH_END)
        _, v_prev = asof(s, PREV_END)
        _, v_ytd = asof(s, YEAR_BASE)
        d_1y, v_1y = asof(s, MONTH_END - pd.DateOffset(years=1))
        r = s.pct_change().dropna()
        r = r[r != 0] if name.endswith("（全收益）") else r                # 全收益指数在节假日有重复的行
        rows.append({"asset": name, "last_date": d_end.date(), "last": v_end, "ret_oct": (v_end / v_prev - 1) * 100,
                     "ret_ytd": (v_end / v_ytd - 1) * 100, "ret_1y": (v_end / v_1y - 1) * 100, "base_1y": d_1y.date(),
                     "vol_60d": r.iloc[-VOL_DAYS:].std() * np.sqrt(ANN) * 100,
                     "complete": d_end == MONTH_END})
    return pd.DataFrame(rows)


def rates_table():
    y = load("cn_us_yields", "日期")
    cols = {"中国 2 年": "中国国债收益率2年", "中国 10 年": "中国国债收益率10年", "中国 30 年": "中国国债收益率30年",
            "美国 10 年": "美国国债收益率10年"}
    rows = {}
    for k, c in cols.items():
        s = y[c].dropna()
        rows[k] = (asof(s, MONTH_END)[1], asof(s, PREV_END)[1])
    rows["中国 10 年减 2 年"] = (rows["中国 10 年"][0] - rows["中国 2 年"][0], rows["中国 10 年"][1] - rows["中国 2 年"][1])
    rows["中美 10 年利差"] = (rows["中国 10 年"][0] - rows["美国 10 年"][0], rows["中国 10 年"][1] - rows["美国 10 年"][1])
    t = pd.DataFrame([{"item": k, "oct_end": a, "sep_end": b, "chg_bp": (a - b) * 100} for k, (a, b) in rows.items()])
    # 判断 2 的依据：过去 36 个月 10 年国债月末变化的分布
    s = y["中国国债收益率10年"].dropna()
    me = s.groupby([s.index.year, s.index.month]).last()
    chg = (me.diff() * 100).dropna().iloc[-36:]
    dist = pd.DataFrame([{"months": len(chg), "p10": chg.quantile(.1), "p25": chg.quantile(.25), "median": chg.median(),
                          "p75": chg.quantile(.75), "p90": chg.quantile(.9), "abs_mean": chg.abs().mean(),
                          "first": f"{chg.index[0][0]}-{chg.index[0][1]:02d}", "last": f"{chg.index[-1][0]}-{chg.index[-1][1]:02d}"}])
    return t, dist


def equity_bond(ser):
    pe = load("hs300_pe").iloc[:, 0].dropna()
    cn10 = load("chinabond_treasury")["10年"].dropna()
    df = pd.DataFrame({"pe": pe, "y10": cn10}).dropna()
    df = df[df.index <= MONTH_END]
    df["erp"] = 100 / df.pe - df.y10                                  # 盈利收益率 − 10 年国债，百分点
    win = df[df.index > MONTH_END - pd.DateOffset(years=5)]
    cur = df.erp.iloc[-1]
    prev = df.erp[df.index <= PREV_END].iloc[-1]
    a = ser["沪深300（全收益）"].pct_change()
    b = ser["中债 7–10 年国债财富"].pct_change()
    j = pd.DataFrame({"stk": a, "bd": b}).dropna()
    j = j[(j.stk != 0) & (j.index <= MONTH_END)]                        # 去掉股市休市、全收益指数重复的日子
    corr_now = j.iloc[-CORR_DAYS:].corr().iloc[0, 1]
    corr_prev = j[j.index <= PREV_END].iloc[-CORR_DAYS:].corr().iloc[0, 1]
    roll = j.stk.rolling(CORR_DAYS).corr(j.bd).dropna()
    roll5 = roll[roll.index > MONTH_END - pd.DateOffset(years=5)]
    return pd.DataFrame([{"date": df.index[-1].date(), "pe": df.pe.iloc[-1], "earnings_yield": 100 / df.pe.iloc[-1],
                          "y10": df.y10.iloc[-1], "erp": cur, "erp_sep": prev, "erp_pct_5y": (win.erp <= cur).mean() * 100,
                          "erp_5y_min": win.erp.min(), "erp_5y_median": win.erp.median(), "erp_5y_max": win.erp.max(),
                          "n_5y": len(win), "corr60": corr_now, "corr60_sep": corr_prev,
                          "corr60_pct_5y": (roll5 <= corr_now).mean() * 100, "corr60_5y_median": roll5.median()}])


def funding():
    f = load("funding_daily")
    rows = []
    for c in ["DR001s", "DR007s"]:
        s = f[c].dropna()
        o, sp = s["2026-10-01":"2026-10-31"], s["2026-09-01":"2026-09-30"]
        rows.append({"rate": c, "oct_mean": o.mean(), "sep_mean": sp.mean(), "chg_bp": (o.mean() - sp.mean()) * 100,
                     "oct_min": o.min(), "oct_max": o.max(), "oct_n": len(o), "last_oct": o.iloc[-1]})
    t = pd.DataFrame(rows)
    pol = f.loc["2026-10-30", ["rate7"]].to_dict()
    pol.update(f.loc["2026-11-02", ["rate7", "rate_on"]].add_suffix("_1102").to_dict())
    return t, pd.DataFrame([pol])


def chart(rt):
    from charts import hbar
    short = {"沪深300（全收益）": "沪深300", "上证50（全收益）": "上证50", "中证500（全收益）": "中证500",
             "中证1000（全收益）": "中证1000", "中债 7–10 年国债财富": "中债 7–10 年", "黄金 ETF": "黄金 ETF",
             "中证商品期货综合": "中证商品", "美元兑人民币中间价": "美元兑人民币", "标普 500": "标普 500"}
    rows = [{"label": short[r.asset], "value": round(float(r.ret_oct), 2)} for r in rt.sort_values("ret_oct", ascending=False).itertuples()]
    save_svg(hbar(rows, title="10 月各资产的收益", unit="%", nd=1, diverging=True,
                  desc="2026-09-30 至 2026-10-30；股票指数用全收益；美元兑人民币中间价为正表示人民币贬值。", axis_note="%"),
             "mon1-returns")


def main():
    ser = series()
    rt = returns_table(ser)
    save_csv(rt, REPORT, "returns")
    rates, dist = rates_table()
    save_csv(rates, REPORT, "rates")
    save_csv(dist, REPORT, "y10_monthly_change_dist")
    eb = equity_bond(ser)
    save_csv(eb, REPORT, "equity_bond")
    fd, pol = funding()
    save_csv(fd, REPORT, "funding")
    save_csv(pol, REPORT, "policy_rates")
    chart(rt)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for name, df in [("returns", rt), ("rates", rates), ("dist", dist), ("equity_bond", eb.T), ("funding", fd), ("policy", pol)]:
            print(f"\n== {name}\n", df.round(4))


if __name__ == "__main__":
    main()
