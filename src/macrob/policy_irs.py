"""第 13 步：用利率互换（FR007）衡量政策意外。运行：uv run python -m macrob.policy_irs

规则（看结果之前定的，写在计划文件第 13 步里）：
- 利率互换：本人从学校 CSMAR 下的成交利率（不是 16:30 的收盘曲线）。2021-04-29 起用加权平均利率，之前用固定利率；
  某天和前后各 2 天的中位数相差 1 个百分点以上，当作录入错误去掉。
- 事件和窗口同第 6 步、补做的调息：公告日前最后一个交易日 → 公告日之后第一个交易日；窗口两端任一天没有成交，这次不算。
- ① 各组事件的 1 年、5 年互换反应（加息乘 −1）；安慰剂：随机窗口避开所有政策事件前后 10 个交易日，2000 次。
- ② Kuttner 分解：只用调利率的事件，意外 = 1 年互换的变化，预期之内 = 调整幅度 − 意外（基点）；
  收益率、沪深300 的反应同时对这两部分回归，异方差稳健 t。
- ③ 数据意外：第一轮和第 12 步的数据意外事件，窗口取到发布日之后第一个交易日（成交利率是全天平均）；
  每 1 个标准差的意外让 1 年互换动多少，异方差稳健 t + 打乱 2000 次；全样本和去掉 2020–2021。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import markets, policy, policy_rates, sources as S
from .credit import CSMAR
from .datalayer import RESULTS

N_PERM, BAD_PP, WIDTH = 2000, 1.0, 2
TENORS = ("1Y", "5Y")
ROUND1_EVENTS = S.UPGRADE_DATA.parent / "results" / "surprise_events.csv"


def irs() -> pd.DataFrame:
    f = pd.read_csv(CSMAR / "FR007_1Y5Y_交易利率_非收盘曲线.csv", parse_dates=["date"])
    out = {}
    for t in TENORS:
        s = f[f["tenor"] == t].set_index("date").sort_index()
        assert not (s["fixed_rate_pct"].notna() & s["weighted_average_rate_pct"].notna()).any()      # 两个字段不重叠
        r = s["weighted_average_rate_pct"].fillna(s["fixed_rate_pct"]).dropna()
        near = pd.concat([r.shift(k) for k in (-2, -1, 1, 2)], axis=1).median(axis=1)
        out[t] = r[~((r - near).abs() > BAD_PP)]
    return pd.DataFrame(out)


def irs_change(rates: pd.DataFrame, days: pd.DatetimeIndex, dates: pd.Series) -> pd.DataFrame:
    """每个日期的窗口（前一个交易日 → 之后第一个交易日）里 1 年、5 年互换的变化，基点。"""
    rows = []
    for d in pd.to_datetime(dates):
        w = policy._window(days, d)
        if w is None:
            rows.append({t: np.nan for t in TENORS})
            continue
        a, b = days[w[0]], days[w[1]]
        rows.append({t: (rates[t].get(b, np.nan) - rates[t].get(a, np.nan)) * 100 for t in TENORS})
    return pd.DataFrame(rows, index=dates.index).rename(columns={t: f"Δ互换{t}(bp)" for t in TENORS})


def policy_events(mk: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    r = policy_rates.rate_events()
    e = pd.concat([r, policy_rates.lpr_events(r).assign(size=lambda x: x["size"] * 100)], ignore_index=True)
    old = policy.events()
    old = old[old["type"].isin(["降准", "升准", "降息（基准）", "加息（基准）"])].assign(分组=lambda x: x["type"], 工具=lambda x: x["type"])
    old.loc[old["type"].str.contains("基准"), "size"] *= 100
    e = pd.concat([e, old[["date", "工具", "size", "direction", "分组"]]], ignore_index=True)
    e["type"] = e["分组"]
    e = policy.reactions(e[e["date"] >= "2010-01-01"].reset_index(drop=True), mk)
    return pd.concat([e, irs_change(rates, mk.index, e["date"])], axis=1)


def groups(e: pd.DataFrame, mk: pd.DataFrame, rates: pd.DataFrame, rng) -> pd.DataFrame:
    days = mk.index
    avoid = set()
    for d in pd.concat([policy.events()["date"], policy_rates.rate_events()["date"]]):
        p = days.searchsorted(d)
        avoid.update(range(p - 10, p + 11))
    cand = np.array([i for i in range(policy.PRE + 1, len(days) - 3) if i not in avoid and days[i] >= rates.index.min()])
    level = {t: rates[t].reindex(days).to_numpy() * 100 for t in TENORS}
    spec = [("首发（加息乘 −1）", e["分组"] == "首发"), ("首发·只看降息", (e["分组"] == "首发") & (e["direction"] < 0)),
            ("跟随（7 天逆回购 / MLF）", e["分组"] == "跟随"), ("LPR 跟随", e["分组"] == "LPR 跟随"), ("LPR 单独", e["分组"] == "LPR 单独"),
            ("发布会", e["分组"] == "发布会"), ("降准", e["分组"] == "降准"), ("升准", e["分组"] == "升准"),
            ("降息（基准）", e["分组"] == "降息（基准）"), ("加息（基准）", e["分组"] == "加息（基准）")]
    rows = []
    for name, m in spec:
        row = {"分组": name}
        for t in TENORS:
            col = f"Δ互换{t}(bp)"
            g = e[m & e[col].notna()]
            sign = -g["direction"].to_numpy()
            x = g[col].to_numpy() * sign
            row[f"{t} 次数"] = len(g)
            row[f"{t} 宽松方向平均(bp)"] = x.mean() if len(x) else np.nan
            row[f"{t} t"] = x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan
            if len(x) > 1:
                fake = np.array([np.nanmean(d) if np.isfinite(d).any() else np.nan
                                 for d in ((level[t][a + WIDTH] - level[t][a]) * sign for a in (rng.choice(cand, size=len(g), replace=False) for _ in range(N_PERM)))])
                fake = fake[~np.isnan(fake)]                    # 抽到的几天都没有成交的，这一次不算
                row[f"{t} 安慰剂p"] = float((np.abs(fake - fake.mean()) >= abs(x.mean() - fake.mean())).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def kuttner(e: pd.DataFrame) -> pd.DataFrame:
    k = e[e["分组"].isin(["首发", "跟随", "LPR 跟随", "LPR 单独", "降息（基准）", "加息（基准）"])].dropna(subset=["Δ互换1Y(bp)"]).copy()
    k["意外(bp)"] = k["Δ互换1Y(bp)"]
    k["预期之内(bp)"] = k["size"] - k["意外(bp)"]
    rows = []
    for col in ("Δ10年(bp)", "Δ2年(bp)", "沪深300(%)"):
        d = k.dropna(subset=[col])
        fit = sm.OLS(d[col], sm.add_constant(d[["预期之内(bp)", "意外(bp)"]])).fit(cov_type="HC1")
        for x in ("预期之内(bp)", "意外(bp)"):
            rows.append({"反应": col, "部分": x, "系数": float(fit.params[x]), "t": float(fit.tvalues[x]), "事件数": len(d), "R²": float(fit.rsquared)})
    return pd.DataFrame(rows), k


def data_surprises(mk: pd.DataFrame, rates: pd.DataFrame, rng) -> pd.DataFrame:
    r1 = pd.read_csv(ROUND1_EVENTS, parse_dates=["date"])
    r1 = r1[r1["indicator"] != "CPI"][["date", "indicator", "surprise"]]
    cpi = pd.read_csv(RESULTS / "surprise_cpi_延长_events.csv", parse_dates=["date"]).assign(indicator="CPI")[["date", "indicator", "surprise"]]
    nbs = pd.read_csv(RESULTS / "surprise_nbs_events_延长.csv", parse_dates=["时间"])
    nbs = nbs[nbs["指标"].isin(["社零", "固投"])].assign(date=lambda x: x["时间"].dt.normalize(), indicator=lambda x: x["指标"])[["date", "indicator", "surprise"]]
    ev = pd.concat([r1, cpi, nbs], ignore_index=True)
    ev = pd.concat([ev, irs_change(rates, mk.index, ev["date"])], axis=1).dropna(subset=["Δ互换1Y(bp)"])
    rows = []
    for name, g in ev.groupby("indicator"):
        covid = g["date"].between("2020-01-01", "2021-12-31")
        for label, sub in (("全样本", g), ("去掉 2020–2021", g[~covid])):
            x, y = sub["surprise"].to_numpy(), sub["Δ互换1Y(bp)"].to_numpy()
            fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC1")
            perm = np.array([np.polyfit(rng.permutation(x), y, 1)[0] for _ in range(N_PERM)])
            rows.append({"指标": name, "样本": label, "每 1 个标准差 Δ互换1Y(bp)": float(fit.params[1]), "t": float(fit.tvalues[1]),
                         "打乱 p": float((np.abs(perm) >= abs(fit.params[1])).mean()), "次数": len(sub)})
    return pd.DataFrame(rows)


def main() -> None:
    mk, rates = markets.daily(), irs()
    rng = np.random.default_rng(0)
    e = policy_events(mk, rates)
    g = groups(e, mk, rates, rng)
    kt, k = kuttner(e)
    ds = data_surprises(mk, rates, rng)
    e.drop(columns=["type", "已有预期"]).to_csv(RESULTS / "policy_irs_events.csv", index=False, encoding="utf-8-sig")
    g.to_csv(RESULTS / "policy_irs_groups.csv", index=False, encoding="utf-8-sig")
    kt.to_csv(RESULTS / "policy_irs_kuttner.csv", index=False, encoding="utf-8-sig")
    ds.to_csv(RESULTS / "surprise_irs.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print(f"利率互换 {rates.index.min():%Y-%m-%d} ~ {rates.index.max():%Y-%m-%d}；去掉的录入错误：1Y {int(irs_raw_count('1Y') - rates['1Y'].count())} 天、5Y {int(irs_raw_count('5Y') - rates['5Y'].count())} 天")
    print("\n① 各组事件的互换反应（宽松方向）：")
    print(g.round(3).to_string(index=False))
    print(f"\n② Kuttner 分解（{len(k)} 次调利率的事件）：")
    print(kt.round(3).to_string(index=False))
    print("\n③ 数据意外对 1 年互换的影响：")
    print(ds.round(3).to_string(index=False))


def irs_raw_count(t: str) -> int:
    f = pd.read_csv(CSMAR / "FR007_1Y5Y_交易利率_非收盘曲线.csv")
    f = f[f["tenor"] == t]
    return int(f["weighted_average_rate_pct"].fillna(f["fixed_rate_pct"]).notna().sum())


if __name__ == "__main__":
    main()
