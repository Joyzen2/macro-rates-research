"""第 18 步：货币政策预期差、国债曲线和资金利率。运行：uv run python -m macrob.policy_funding

规则（看结果之前定的，写在计划文件第 18 步里）：
- 事件、意外、预期之内同第 13 步的 Kuttner 分解（40 次调利率的事件；意外 = 窗口内 1 年互换的变化，预期之内 = 调整幅度 − 意外）。
- 曲线：中债国债 8 个期限的窗口变化，同时对意外、预期之内回归，异方差稳健 t。
- 资金利率：公告后第 1–5 个交易日均值 − 公告前第 1–5 个交易日均值（公告当天不算）；同样回归，并检验两个系数相等。
  稳健性：中位数、前后各 20 个交易日；全部事件和去掉 LPR 两组的都报。
  看完结果后加的（标明是修正）：只看 2015 年以后的事件，因为 2015 年之前资金利率的日度波动大得多；5 日窗口做留一法。
- 路径：首发、跟随、LPR 跟随三组，FR007 相对公告前 5 日均值的平均路径（宽松方向，加息乘 −1）。
- 分组的 5 日变化做安慰剂：随机窗口避开所有政策事件前后 10 个交易日，2000 次。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import markets, policy, policy_rates
from .curve import COLS, curve_daily
from .datalayer import RESULTS
from .probe_util import RAW

GROUPS_K = ["首发", "跟随", "LPR 跟随", "LPR 单独", "降息（基准）", "加息（基准）"]
NO_LPR = ["首发", "跟随", "降息（基准）", "加息（基准）"]
FUNDING = ["FR007", "FR001", "FDR007", "FDR001"]
N_PERM, AVOID = 2000, 10
POST = pd.Timestamp("2015-01-01")


def kuttner_events() -> pd.DataFrame:
    e = pd.read_csv(RESULTS / "policy_irs_events.csv", parse_dates=["date"], encoding="utf-8-sig")
    k = e[e["分组"].isin(GROUPS_K)].dropna(subset=["Δ互换1Y(bp)"]).copy()
    k["意外(bp)"] = k["Δ互换1Y(bp)"]
    k["预期之内(bp)"] = k["size"] - k["意外(bp)"]
    return k.sort_values("date").reset_index(drop=True)


def fixings() -> pd.DataFrame:
    return pd.read_csv(RAW / "repo_rate_hist.csv", parse_dates=["date"]).set_index("date").sort_index()


def around(s: pd.Series, d: pd.Timestamp, n: int, how: str = "mean") -> float:
    """公告后第 1–n 个交易日 − 公告前第 1–n 个交易日（公告当天不算），基点。"""
    x = s.dropna()
    pre, post = x[x.index < d].iloc[-n:], x[x.index > d].iloc[:n]
    if len(pre) < n or len(post) < n:
        return np.nan
    return float((getattr(post, how)() - getattr(pre, how)()) * 100)


def regress(y: pd.Series, k: pd.DataFrame) -> dict:
    d = pd.concat([y.rename("y"), k[["意外(bp)", "预期之内(bp)"]]], axis=1).dropna()
    fit = sm.OLS(d["y"], sm.add_constant(d[["意外(bp)", "预期之内(bp)"]])).fit(cov_type="HC1")
    eq = fit.t_test(np.array([0, 1, -1]))                    # 意外和预期之内的系数相等
    return {"意外系数": float(fit.params["意外(bp)"]), "意外t": float(fit.tvalues["意外(bp)"]),
            "预期之内系数": float(fit.params["预期之内(bp)"]), "预期之内t": float(fit.tvalues["预期之内(bp)"]),
            "相等检验p": float(eq.pvalue), "事件数": int(len(d)), "R²": float(fit.rsquared)}


def curve_table(k: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    c = curve_daily()
    rows = []
    for t in COLS:
        s = c[t].reindex(days).ffill()
        ch = []
        for d in k["date"]:
            w = policy._window(days, d)
            ch.append(np.nan if w is None else (s.iloc[w[1]] - s.iloc[w[0]]) * 100)
        rows.append({"期限": t, **regress(pd.Series(ch, index=k.index), k)})
    return pd.DataFrame(rows)


def funding_table(k: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    post = k[k["date"] >= POST]
    for sample, sub in (("全部", k), ("去掉 LPR", k[k["分组"].isin(NO_LPR)]), ("2015 年后", post), ("2015 年后去掉 LPR", post[post["分组"].isin(NO_LPR)])):
        for name in FUNDING:
            for label, n, how in (("5 日均值", 5, "mean"), ("5 日中位数", 5, "median"), ("20 日均值", 20, "mean")):
                y = pd.Series([around(fx[name], d, n, how) for d in sub["date"]], index=sub.index)
                rows.append({"样本": sample, "资金利率": name, "窗口": label, **regress(y, sub)})
    return pd.DataFrame(rows)


def leave_one_out(k: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """2015 年后、去掉 LPR 的 5 日窗口只有十几次事件，逐次去掉一次，看系数和 t 的范围。"""
    sub = k[(k["date"] >= POST) & k["分组"].isin(NO_LPR)]
    rows = []
    for name in ("FDR007", "FDR001"):
        y = pd.Series([around(fx[name], d, 5) for d in sub["date"]], index=sub.index)
        fits = [regress(y.drop(i), sub.drop(i)) for i in y.dropna().index]
        rows.append({"资金利率": name, "事件数": int(y.notna().sum()), "意外系数最小": min(f["意外系数"] for f in fits), "意外系数最大": max(f["意外系数"] for f in fits),
                     "意外t最小": min(f["意外t"] for f in fits), "意外t最大": max(f["意外t"] for f in fits)})
    return pd.DataFrame(rows)


def _avoid(days: pd.DatetimeIndex) -> set:
    dates = pd.concat([policy.events()["date"], policy_rates.rate_events()["date"]])
    out = set()
    for d in dates:
        p = days.searchsorted(d)
        out.update(range(p - AVOID, p + AVOID + 1))
    return out


def group_table(k: pd.DataFrame, fx: pd.DataFrame, rng) -> pd.DataFrame:
    """分组的 5 日变化（宽松方向）和安慰剂。"""
    s = fx["FR007"].dropna()
    days = s.index
    avoid = _avoid(days)
    cand = np.array([i for i in range(6, len(days) - 6) if i not in avoid])
    v = s.to_numpy()
    rolled = np.array([v[i + 1:i + 6].mean() - v[i - 5:i].mean() for i in cand]) * 100
    rows = []
    groups = [("首发·只看降息", k[(k["分组"] == "首发") & (k["direction"] < 0)]), ("首发（加息乘 −1）", k[k["分组"] == "首发"]),
              ("跟随（7 天逆回购 / MLF）", k[k["分组"] == "跟随"]), ("LPR 跟随", k[k["分组"] == "LPR 跟随"]),
              ("基准利率调整（加息乘 −1）", k[k["分组"].isin(["降息（基准）", "加息（基准）"])])]
    for name, g in groups:
        sign = -g["direction"].to_numpy()
        x = np.array([around(s, d, 5) for d in g["date"]]) * sign
        x = x[~np.isnan(x)]
        size = (g["size"].to_numpy() * sign).mean()
        fake = np.array([np.mean(rng.choice(rolled, size=len(x), replace=False) * rng.choice(sign, size=len(x))) for _ in range(N_PERM)])
        rows.append({"分组": name, "次数": len(x), "平均调整幅度(bp)": float(size), "FR007 5日变化(bp)": float(x.mean()),
                     "t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else np.nan,
                     "安慰剂p": float((np.abs(fake - fake.mean()) >= abs(x.mean() - fake.mean())).mean()),
                     "意外(bp)": float((g["意外(bp)"].to_numpy() * sign).mean()), "Δ10年(bp)": float((g["Δ10年(bp)"].to_numpy() * sign).mean())})
    return pd.DataFrame(rows)


def paths(k: pd.DataFrame, fx: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    s = fx["FR007"].dropna()
    days = s.index
    out = {}
    for name, g in (("首发", k[k["分组"] == "首发"]), ("跟随", k[k["分组"] == "跟随"]), ("LPR 跟随", k[k["分组"] == "LPR 跟随"])):
        rows = []
        for d, sgn in zip(g["date"], -g["direction"]):
            p = days.searchsorted(d)
            if p - span < 0 or p + span >= len(days):
                continue
            seg = s.iloc[p - span:p + span + 1].to_numpy()
            base = seg[span - 5:span].mean()
            rows.append((seg - base) * 100 * sgn)
        out[name] = np.mean(rows, axis=0)
    return pd.DataFrame(out, index=pd.RangeIndex(-span, span + 1, name="相对公告日的交易日"))


def main() -> None:
    mk = markets.daily()
    k, fx = kuttner_events(), fixings()
    rng = np.random.default_rng(0)
    ct = curve_table(k, mk.index)
    ft = funding_table(k, fx)
    gt = group_table(k, fx, rng)
    pt = paths(k, fx)
    lo = leave_one_out(k, fx)
    ct.to_csv(RESULTS / "report1_curve.csv", index=False, encoding="utf-8-sig")
    ft.to_csv(RESULTS / "report1_funding.csv", index=False, encoding="utf-8-sig")
    gt.to_csv(RESULTS / "report1_groups.csv", index=False, encoding="utf-8-sig")
    pt.to_csv(RESULTS / "report1_paths.csv", encoding="utf-8-sig")
    lo.to_csv(RESULTS / "report1_leave_one_out.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    print("事件", len(k), k["分组"].value_counts().to_dict(), "| 工具", k["工具"].value_counts().to_dict(), "| 加息", int((k["direction"] > 0).sum()))
    print("\n① 曲线"); print(ct.round(3).to_string(index=False))
    print("\n② 资金利率"); print(ft.round(3).to_string(index=False))
    print("\n③ 分组"); print(gt.round(3).to_string(index=False))
    print("\n留一法"); print(lo.round(3).to_string(index=False))
    print("\n④ 路径（选几个点）"); print(pt.loc[[-20, -10, -5, -1, 0, 1, 2, 3, 5, 10, 20]].round(1).to_string())


if __name__ == "__main__":
    main()
