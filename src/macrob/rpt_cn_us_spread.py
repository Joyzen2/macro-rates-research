"""第三篇：中美利差。美债收益率的变动会不会传到中国国债，传多少，什么时候强，汇率渠道怎么样。

uv run python -m macrob.rpt_cn_us_spread
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .rpt_common import load, month_ends, nw_ols, save_csv, save_svg, shift_placebo, spearman

REPORT = "cn-us-yield-spread"
START = pd.Timestamp("2010-01-01")
TENORS = {"2年": ("中国国债收益率2年", "美国国债收益率2年"), "10年": ("中国国债收益率10年", "美国国债收益率10年"),
          "30年": ("中国国债收益率30年", "美国国债收益率30年")}
PERIODS = [("2010–2014", "2010-01-01", "2014-12-31"), ("2015–2017", "2015-01-01", "2017-12-31"),
           ("2018–2019", "2018-01-01", "2019-12-31"), ("2020–2021", "2020-01-01", "2021-12-31"),
           ("2022–2023", "2022-01-01", "2023-12-31"), ("2024 起", "2024-01-01", "2026-12-31")]
OUTLIER_BP = 50
# 逐条看过、确认是数据错误的单日变化（日期, 列名）。目前为空：唯一超过 50bp 的是美国 2 年 2023-03-13（−57bp），
# 前后几天连续下行（4.90、4.60、4.03、4.20、3.93），不是单日尖峰，没有理由认定是错的，保留。
CONFIRMED_ERRORS: list[tuple[str, str]] = []


def changes(s: pd.Series) -> pd.Series:
    """各自交易日上的日变化，bp。"""
    s = s.dropna()
    return s.diff() * 100


def match_prev_us(cn_dates: pd.DatetimeIndex, us_dates: pd.DatetimeIndex) -> np.ndarray:
    """中国第 c 个交易日配日期严格早于 c 的最后一个美国交易日；没有则为 NaT。"""
    pos = np.searchsorted(us_dates.values, cn_dates.values, side="left") - 1
    out = np.full(len(cn_dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = pos >= 0
    out[ok] = us_dates.values[pos[ok]]
    return out


def daily_pairs(y: pd.DataFrame, cn_col: str, us_col: str) -> pd.DataFrame:
    dcn, dus = changes(y[cn_col]), changes(y[us_col])
    for d, col in CONFIRMED_ERRORS:
        if col == cn_col:
            dcn = dcn.drop(pd.Timestamp(d), errors="ignore")
        if col == us_col:
            dus = dus.drop(pd.Timestamp(d), errors="ignore")
    us_d = match_prev_us(dcn.index, dus.index)
    df = pd.DataFrame({"d_cn": dcn.values, "us_date": us_d}, index=dcn.index)
    df["d_us"] = dus.reindex(pd.DatetimeIndex(df.us_date)).values
    df = df[df.index >= START].dropna()
    return df


def outliers(y: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (cn_col, us_col) in TENORS.items():
        for col in (cn_col, us_col):
            s = y[col].dropna()
            d = s.diff() * 100
            for t, v in d[(d.abs() > OUTLIER_BP) & (d.index >= START)].items():
                i = s.index.get_loc(t)
                rows.append({"series": col, "date": t.date(), "change_bp": v,
                             "levels_around": " ".join(f"{x:.3f}" for x in s.iloc[max(0, i - 2): i + 3]),
                             "confirmed_error": (str(t.date()), col) in CONFIRMED_ERRORS})
    return pd.DataFrame(rows, columns=["series", "date", "change_bp", "levels_around", "confirmed_error"])


def weekly_pairs(y, cn_col, us_col):
    cn = y[cn_col].dropna().resample("W-FRI").last()
    us = y[us_col].dropna().resample("W-FRI").last()
    df = pd.DataFrame({"d_cn": cn.diff() * 100, "d_us": us.diff() * 100})
    return df[df.index >= START + pd.Timedelta(days=7)].dropna()          # 第一周的起点也在 2010 年里


def passthrough(y):
    rows = []
    for name, (cn_col, us_col) in TENORS.items():
        d = daily_pairs(y, cn_col, us_col)
        r = nw_ols(d.d_cn, d.d_us, 5)
        rows.append({"tenor": name, "freq": "日度", "start": d.index.min().date(), "end": d.index.max().date(), **r,
                     "corr": d[["d_cn", "d_us"]].corr().iloc[0, 1]})
        w = weekly_pairs(y, cn_col, us_col)
        r = nw_ols(w.d_cn, w.d_us, 5)
        rows.append({"tenor": name, "freq": "周度", "start": w.index.min().date(), "end": w.index.max().date(), **r,
                     "corr": w[["d_cn", "d_us"]].corr().iloc[0, 1]})
    return pd.DataFrame(rows)


def by_period(d):
    rows = []
    for name, a, b in PERIODS:
        h = d.loc[a:b]
        r = nw_ols(h.d_cn, h.d_us, 5)
        rows.append({"period": name, "start": h.index.min().date(), "end": h.index.max().date(), **r,
                     "sd_cn": h.d_cn.std(), "sd_us": h.d_us.std()})
    return pd.DataFrame(rows)


def rolling_beta(d, window=250):
    x, yv = d.d_us, d.d_cn
    cov = x.rolling(window).cov(yv)
    var = x.rolling(window).var()
    return (cov / var).dropna().rename("beta")


def fomc_test(d, fomc):
    dates = set(pd.to_datetime(fomc.us_date))
    d = d.copy()
    d["fomc"] = d.us_date.isin(dates).astype(float)
    d["inter"] = d.fomc * d.d_us
    X = sm.add_constant(d[["d_us", "fomc", "inter"]])
    r = sm.OLS(d.d_cn, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    f = d[d.fomc == 1]
    rf = nw_ols(f.d_cn, f.d_us, 1)
    summary = pd.DataFrame([{
        "n_all": len(d), "n_fomc": int(d.fomc.sum()), "b_normal": r.params["d_us"], "t_b_normal": r.tvalues["d_us"],
        "fomc_dummy": r.params["fomc"], "t_dummy": r.tvalues["fomc"], "interaction": r.params["inter"],
        "t_interaction": r.tvalues["inter"], "b_fomc": r.params["d_us"] + r.params["inter"],
        "b_fomc_subsample": rf["b"], "t_fomc_subsample": rf["t_b"],
        "sd_us_fomc": f.d_us.std(), "sd_us_other": d[d.fomc == 0].d_us.std(),
        "mean_abs_cn_fomc": f.d_cn.abs().mean(), "mean_abs_cn_other": d[d.fomc == 0].d_cn.abs().mean()}])
    return summary, f.reset_index(names="cn_date")


def fx_test(y, parity):
    cn, us = y["中国国债收益率10年"].dropna(), y["美国国债收益率10年"].dropna()
    p = parity.iloc[:, 0].dropna()
    m = lambda s: s.groupby([s.index.year, s.index.month]).last()   # noqa: E731
    df = pd.DataFrame({"cn": m(cn), "us": m(us), "parity": m(p)})
    df.index = pd.PeriodIndex([pd.Period(year=a, month=b, freq="M") for a, b in df.index])
    df = df[df.index >= pd.Period("2010-01", "M")].dropna()
    df = df[df.index <= pd.Period("2026-10", "M")]
    df["spread"] = (df.cn - df.us) * 100                                     # bp
    df["d_parity_next"] = (df.parity.shift(-1) / df.parity - 1) * 100       # %，正数是人民币贬值
    df["d_parity"] = (df.parity / df.parity.shift(1) - 1) * 100
    df["d_spread"] = df.spread.diff()
    df["d_cn"] = df.cn.diff() * 100
    df["d_us"] = df.us.diff() * 100
    rows = []
    a = df.dropna(subset=["d_parity_next"])
    rho, pp, nsh = shift_placebo(a.spread.values, a.d_parity_next.values)
    r = nw_ols(a.d_parity_next, a.spread, 1)
    rows.append({"test": "月末利差 → 下月中间价变化", "n": len(a), "start": str(a.index.min()), "end": str(a.index.max()),
                 "spearman": rho, "b": r["b"], "t_nw": r["t_b"], "r2": r["r2"], "placebo_p": pp, "n_shifts": nsh})
    for name, xc, yc in [("同期：利差变化 → 中间价变化", "d_spread", "d_parity"),
                         ("同期：美国 10 年变化 → 中间价变化", "d_us", "d_parity"),
                         ("同期：美国 10 年变化 → 中国 10 年变化", "d_us", "d_cn")]:
        b = df.dropna(subset=[xc, yc])
        rho, pp, nsh = shift_placebo(b[xc].values, b[yc].values)
        r = nw_ols(b[yc], b[xc], 1)
        rows.append({"test": name, "n": len(b), "start": str(b.index.min()), "end": str(b.index.max()), "spearman": rho,
                     "b": r["b"], "t_nw": r["t_b"], "r2": r["r2"], "placebo_p": pp, "n_shifts": nsh,
                     "pearson": b[[xc, yc]].corr().iloc[0, 1]})
    return pd.DataFrame(rows), df.reset_index(names="month")


def level(y):
    cn, us = y["中国国债收益率10年"].dropna(), y["美国国债收益率10年"].dropna()
    us_on = us.reindex(cn.index, method="ffill")                           # 中国交易日，用当时最新的美国收盘
    s = ((cn - us_on) * 100).dropna()
    s = s[s.index >= START]
    w = s.resample("W-FRI").last().dropna()
    lv = pd.DataFrame([{"date": s.index[-1].date(), "cn10": cn.iloc[-1], "us10": us_on.iloc[-1], "spread_bp": s.iloc[-1],
                        "min_bp": s.min(), "min_date": s.idxmin().date(), "max_bp": s.max(), "max_date": s.idxmax().date(),
                        "first_negative": s[s < 0].index.min().date() if (s < 0).any() else None,
                        "share_negative_since_2022": float((s.loc["2022":] < 0).mean())}])
    return s, w, lv


def charts(weekly, roll, per):
    from charts import INK, S, hbar, line_chart
    dates = list(weekly.index)
    save_svg(line_chart(dates, [("中国减美国 10 年", (weekly / 100).round(3).tolist(), INK)],
                        title="中美 10 年国债利差", ylo=-4, yhi=3, yticks=[-4, -3, -2, -1, 0, 1, 2, 3], ylabel="%",
                        ref=0.0, year_step=2, desc="周五值，%。2010 年 1 月至 2026 年 10 月。"), "rpt4-spread")
    r = roll[roll.index >= "2011-01-01"].resample("W-FRI").last().dropna()
    save_svg(line_chart(list(r.index), [("滚动 250 日的传导系数 × 100", (r * 100).round(2).tolist(), S[1])],
                        title="美债 10 年每变 100bp，中国 10 年第二天变多少 bp", ylo=-5, yhi=20, yticks=[-5, 0, 5, 10, 15, 20],
                        ylabel="bp", ref=0.0, year_step=2, desc="滚动 250 个交易日的回归斜率乘以 100，周五值。"), "rpt4-rolling")
    rows = [{"label": p.period, "value": float(p.b) * 100, "mark": "*" if abs(p.t_b) >= 1.96 else ""} for p in per.itertuples()]
    for row in rows:
        row["color"] = INK if row["mark"] else "#C5D0DB"
    save_svg(hbar(rows, title="分时期的传导系数（10 年，日度）", nd=1, signed=True,
                  desc="美债 10 年每变 100bp，中国 10 年第二天变多少 bp。星号表示 Newey-West t 的绝对值不小于 1.96，淡色表示不显著。",
                  axis_note="bp / 100bp"), "rpt4-periods")


def main():
    y = load("cn_us_yields", "日期")
    y = y[y.index >= "2009-11-01"]
    ol = outliers(y)
    save_csv(ol, REPORT, "outliers")
    pt = passthrough(y)
    save_csv(pt, REPORT, "passthrough")
    d10 = daily_pairs(y, *TENORS["10年"])
    save_csv(d10.reset_index(names="cn_date"), REPORT, "daily_10y")
    per = by_period(d10)
    save_csv(per, REPORT, "periods")
    roll = rolling_beta(d10)
    save_csv(roll.rename_axis("cn_date").reset_index(), REPORT, "rolling_beta")
    fs, fe = fomc_test(d10, load("fomc_dates", "beijing_time").reset_index())
    save_csv(fs, REPORT, "fomc")
    save_csv(fe, REPORT, "fomc_events")
    fx, fxm = fx_test(y, load("usdcny_parity"))
    save_csv(fx, REPORT, "fx")
    save_csv(fxm, REPORT, "fx_monthly")
    s, w, lv = level(y)
    save_csv(lv, REPORT, "level")
    save_csv(w.rename("spread_bp").rename_axis("week").reset_index(), REPORT, "spread_weekly")
    charts(w, roll, per)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for name, df in [("outliers", ol), ("passthrough", pt), ("periods", per), ("fomc", fs.T), ("fx", fx), ("level", lv.T)]:
            print(f"\n== {name}\n", df.round(4))
        print(roll.describe(), roll.iloc[-1])


if __name__ == "__main__":
    main()
