"""第 11 步：收益率曲线。运行：uv run python -m macrob.curve

规则（看结果之前定的，写在计划文件第 11 步里）：
- 数据：中债国债收益率曲线，8 个期限，2006-03 起日度，取月末。
- 因子：Nelson-Siegel，衰减参数固定 0.0609（期限按月），每期最小二乘；斜率 = 长端减短端（−β2）。
- 问题一：三个因子和增长、通胀维度、FR007 的相关（2011 起）；另报 12 个月变化的相关（序列持续性强，水平相关容易虚高）。
- 问题二：10 年 − 3 个月利差 → 之后 12 个月 PMI、增长维度的变化：秩相关、Newey-West t（滞后 12）、平移安慰剂 2000 次。
- 问题三：10 年 − 1 年利差 → 之后 12 个月 10 年国债超额收益（合成总收益 − 1 年期收益率）：
  全样本 Newey-West t、平移安慰剂；样本外 R²（2012 起扩张窗口，对照历史均值）。
- 问题四：政策事件窗口内三个因子的变化（宽松方向：加息乘 −1），随机非事件窗口做安慰剂。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import policy, policy_rates
from .allocation import daily_levels
from .datalayer import RESULTS, raw_panel
from .probe_util import RAW
from .state import dimensions

COLS = ["3月", "6月", "1年", "3年", "5年", "7年", "10年", "30年"]
TAU = np.array([3, 6, 12, 36, 60, 84, 120, 360], dtype=float)
LAM, N_PERM, H = 0.0609, 2000, 12
OOS_START = pd.Timestamp("2012-01-31")


def curve_daily() -> pd.DataFrame:
    c = pd.concat([pd.read_csv(RAW / f"chinabond_curve_{y}.csv") for y in range(2006, 2027)])
    c = c[c["曲线名称"] == "中债国债收益率曲线"].assign(日期=lambda d: pd.to_datetime(d["日期"]))
    return c.drop_duplicates("日期").set_index("日期").sort_index()[COLS].astype(float)


def ns_factors(curve: pd.DataFrame) -> pd.DataFrame:
    x = LAM * TAU
    l2 = (1 - np.exp(-x)) / x
    X = np.column_stack([np.ones_like(x), l2, l2 - np.exp(-x)])
    B, *_ = np.linalg.lstsq(X, curve.to_numpy().T, rcond=None)
    f = pd.DataFrame(B.T, index=curve.index, columns=["水平", "b2", "曲率"])
    f["斜率"] = -f["b2"]
    fit = curve.to_numpy() - B.T @ X.T
    f["拟合误差(bp)"] = np.sqrt((fit ** 2).mean(axis=1)) * 100
    return f[["水平", "斜率", "曲率", "拟合误差(bp)"]]


def nw_t(y: pd.Series, x: pd.Series, lags: int = H) -> tuple[float, float]:
    r = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params.iloc[1]), float(r.tvalues.iloc[1])


def shift_placebo(x: np.ndarray, y: np.ndarray, stat, rng) -> float:
    real = stat(x, y)
    fake = np.array([stat(np.roll(x, k), y) for k in rng.integers(H, len(x) - H, size=N_PERM)])
    return float((np.abs(fake) >= abs(real)).mean())


def spearman(x, y) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def slope_beta(x, y) -> float:
    return float(np.polyfit(x, y, 1)[0])


def q1_meaning(fm: pd.DataFrame, dims: pd.DataFrame, fr007: pd.Series) -> pd.DataFrame:
    d = pd.concat([fm[["水平", "斜率", "曲率"]], dims, fr007.rename("FR007")], axis=1).dropna()
    d = d[d.index >= "2011-01-31"]
    lv = d.corr().loc[["水平", "斜率", "曲率"], ["增长", "通胀", "FR007"]]
    ch = (d - d.shift(H)).dropna().corr().loc[["水平", "斜率", "曲率"], ["增长", "通胀", "FR007"]]
    return pd.concat({"水平相关": lv, "12 个月变化的相关": ch}, axis=1)


def q2_leads(curve_m: pd.DataFrame, pmi: pd.Series, dims: pd.DataFrame, rng) -> pd.DataFrame:
    spread = (curve_m["10年"] - curve_m["3月"]).rename("x")
    rows = []
    for name, target in (("之后 12 个月 PMI 变化", pmi.shift(-H) - pmi), ("之后 12 个月增长维度变化", dims["增长"].shift(-H) - dims["增长"])):
        d = pd.concat([spread, target.rename("y")], axis=1).dropna()
        b, t = nw_t(d["y"], d["x"])
        rows.append({"目标": name, "月数": len(d), "起": f"{d.index.min():%Y-%m}", "秩相关": spearman(d["x"], d["y"]), "斜率系数": b,
                     "Newey-West t": t, "平移安慰剂 p": shift_placebo(d["x"].to_numpy(), d["y"].to_numpy(), spearman, rng)})
    return pd.DataFrame(rows)


def q3_excess(curve_m: pd.DataFrame, tr_m: pd.Series, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = (curve_m["10年"] - curve_m["1年"]).rename("x")
    xr = ((tr_m.shift(-H) / tr_m - 1) * 100 - curve_m["1年"]).rename("y")          # 百分比
    d = pd.concat([x, xr], axis=1).dropna()
    b, t = nw_t(d["y"], d["x"])
    r2 = float(sm.OLS(d["y"], sm.add_constant(d["x"])).fit().rsquared)
    p = shift_placebo(d["x"].to_numpy(), d["y"].to_numpy(), slope_beta, rng)
    preds = []
    for T in d.index[d.index >= OOS_START]:
        past = d[d.index <= T - pd.DateOffset(months=H)]
        if len(past) < 60:
            continue
        bb = np.polyfit(past["x"], past["y"], 1)
        preds.append({"date": T, "y": d.loc[T, "y"], "模型": bb[0] * d.loc[T, "x"] + bb[1], "历史均值": past["y"].mean()})
    pr = pd.DataFrame(preds).set_index("date")
    r2_os = 1 - ((pr["y"] - pr["模型"]) ** 2).sum() / ((pr["y"] - pr["历史均值"]) ** 2).sum()
    summary = pd.DataFrame([{"月数": len(d), "起": f"{d.index.min():%Y-%m}", "止": f"{d.index.max():%Y-%m}", "系数（每 1 个百分点利差）": b,
                             "Newey-West t": t, "全样本 R²": r2, "平移安慰剂 p": p, "样本外月数": len(pr), "样本外 R²": float(r2_os)}])
    return summary, pr


def q4_events(curve_d: pd.DataFrame, fd: pd.DataFrame, rng) -> pd.DataFrame:
    old = policy.events()
    rates = policy_rates.rate_events()
    lpr = policy_rates.lpr_events(rates)
    groups = {"降息（基准）": old[old["type"] == "降息（基准）"], "加息（基准）": old[old["type"] == "加息（基准）"],
              "降准": old[old["type"] == "降准"], "首发降息（7 天逆回购 / MLF）": rates[(rates["分组"] == "首发") & (rates["direction"] < 0)],
              "跟随（7 天逆回购 / MLF，宽松方向）": rates[rates["分组"] == "跟随"], "LPR 跟随": lpr[lpr["分组"] == "LPR 跟随"]}
    days = fd.index
    all_dates = pd.concat([old["date"], rates["date"]])
    avoid = set()
    for d0 in all_dates:
        p0 = days.searchsorted(d0)
        avoid.update(range(p0 - 10, p0 + 11))
    cand = np.array([i for i in range(1, len(days) - 3) if i not in avoid])
    rows = []
    for name, e in groups.items():
        win = []
        for _, r in e.iterrows():
            a = days.searchsorted(r["date"], side="left") - 1
            b = days.searchsorted(r["date"], side="right")
            if a >= 0 and b < len(days):
                win.append((a, b, -int(r["direction"])))
        if not win:
            continue
        row = {"事件": name, "次数": len(win)}
        sign = np.array([w[2] for w in win], dtype=float)
        for col in ("水平", "斜率", "曲率"):
            v = fd[col].to_numpy() * 100
            x = np.array([v[b] - v[a] for a, b, _ in win]) * sign
            row[f"Δ{col}(bp)"] = x.mean()
            row[f"Δ{col} t"] = x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan
            fake = np.array([np.mean((v[(aa := rng.choice(cand, size=len(win), replace=False)) + 2] - v[aa]) * sign) for _ in range(N_PERM)])
            row[f"Δ{col} 安慰剂p"] = float((np.abs(fake - fake.mean()) >= abs(x.mean() - fake.mean())).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(0)
    cd = curve_daily()
    fd = ns_factors(cd)
    cm = cd.resample("ME").last()
    fm = fd.resample("ME").last()
    fm.to_csv(RESULTS / "curve_factors_monthly.csv", encoding="utf-8-sig")
    print(f"曲线 {cd.index.min().date()} ~ {cd.index.max().date()}，{len(cd)} 个交易日；Nelson-Siegel 拟合误差中位数 {fd['拟合误差(bp)'].median():.1f} bp")
    raw = raw_panel()
    dims = dimensions()
    fr007 = raw["fr007"]
    pd.set_option("display.width", 240)
    t1 = q1_meaning(fm, dims, fr007)
    t1.to_csv(RESULTS / "curve_q1_meaning.csv", encoding="utf-8-sig")
    print("\n问题一：因子和宏观的相关（2011 起）\n", t1.round(2).to_string())
    t2 = q2_leads(cm, raw["pmi"], dims, rng)
    t2.to_csv(RESULTS / "curve_q2_slope_leads.csv", index=False, encoding="utf-8-sig")
    print("\n问题二：10 年 − 3 个月利差领先经济吗\n", t2.round(3).to_string(index=False))
    tr_m = daily_levels()["债"].resample("ME").last()
    t3, pr = q3_excess(cm, tr_m, rng)
    t3.to_csv(RESULTS / "curve_q3_excess_return.csv", index=False, encoding="utf-8-sig")
    pr.to_csv(RESULTS / "curve_q3_oos_predictions.csv", encoding="utf-8-sig")
    print("\n问题三：10 年 − 1 年利差 → 之后 12 个月 10 年国债超额收益\n", t3.round(3).to_string(index=False))
    t4 = q4_events(cd, fd, rng)
    t4.to_csv(RESULTS / "curve_q4_policy_events.csv", index=False, encoding="utf-8-sig")
    print("\n问题四：政策事件窗口内曲线因子的变化（宽松方向）\n", t4.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
