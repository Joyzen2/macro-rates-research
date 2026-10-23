"""第 10 步：GDP 实时预测（nowcasting）。运行：uv run python -m macrob.nowcast

规则（看结果之前定的，写在计划文件第 10 步里）：
- 目标：季度 GDP 实际同比的首次公布值（金十，2010Q4–2025Q2）；更早和 2025Q3 以后用统计局现在的数。
- 月度指标：17 个经济活动指标；2020-01 至 2021-12 整段不用，2020Q1–2021Q4 不参与回归和评价。
- 时点：每个季度第 1、2、3 个月的月末，只用当时已经公布的数（按数据层的滞后规则）。
- 第一步：DynamicFactorMQ（1 个因子、因子 AR(2)、个体误差 AR(1)、EM），每个时点重估，平滑补缺口并外推到季末；
  当时观测少于 24 个月的指标不放进模型。
- 第二步：GDP 同比 = a + b × 季度内因子的平均 + c × 上季度 GDP 同比，只用当时已公布的季度。
- 对照：上季度不变、AR(1)、工业增加值桥接、金十一致预期。评价区间 2013Q1–2025Q2（去掉疫情 8 个季度）。
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .datalayer import REGISTRY, RESULTS, ROOT, available_at, raw_panel
from .probe_util import cached
from .sources import UPGRADE_DATA

ACT = ["ip_yoy", "pmi", "pmi_prod", "pmi_new", "power_yoy", "fai_ytd", "re_inv_ytd", "re_sales_ytd", "retail_yoy", "autos_yoy",
       "services_yoy", "export_yoy", "import_yoy", "pmi_export", "loans_12m", "m1_yoy", "m2_yoy"]
LAG = {i.key: i.lag for i in REGISTRY}
COVID_M = (pd.Timestamp("2020-01-31"), pd.Timestamp("2021-12-31"))
COVID_Q = set(pd.period_range("2020Q1", "2021Q4", freq="Q"))
FIRST_Q, LAST_EVAL_Q, LIVE_Q = pd.Period("2013Q1", "Q"), pd.Period("2025Q2", "Q"), pd.Period("2026Q3", "Q")
START, MIN_OBS = pd.Timestamp("2005-01-31"), 24
UPGRADE = UPGRADE_DATA
CN_Q = {"一": 1, "二": 2, "三": 3, "四": 4}


def gdp_releases() -> pd.DataFrame:
    """金十：每次发布的首次公布值和一致预期，按所属季度。"""
    g = pd.read_csv(UPGRADE / "consensus_macro_china_gdp_yearly.csv", parse_dates=["日期"]).dropna(subset=["今值"])
    g["q"] = g["日期"].dt.to_period("Q") - 1
    g = g.drop_duplicates("q", keep="first").set_index("q")
    return pd.DataFrame({"首次公布": pd.to_numeric(g["今值"]), "一致预期": pd.to_numeric(g["预测值"], errors="coerce"), "发布日": g["日期"]})


def gdp_nbs() -> pd.Series:
    """统计局现在的数：国内生产总值指数（上年同期 = 100）当季值 − 100。"""
    import akshare as ak
    d = cached("nbs_gdp_index_q", lambda: ak.macro_china_nbs_nation(kind="季度数据", path="国民经济核算 > 国内生产总值指数", period="2005-")
               .reset_index().rename(columns={"index": "指标"}))
    d = d.set_index(d.columns[0])
    row = d.loc[[i for i in d.index if re.search(r"国内生产总值指数.*当季", str(i))][0]]
    out = {}
    for col, v in row.items():
        m = re.match(r"(\d{4})年第(.)季度", str(col))
        if m and pd.notna(v):
            out[pd.Period(f"{m.group(1)}Q{CN_Q[m.group(2)]}", "Q")] = float(v) - 100
    return pd.Series(out).sort_index()


def gdp_target() -> pd.Series:
    """训练和外推用的 GDP：有首次公布值就用首次公布值，否则用统计局现在的数。"""
    nbs, rel = gdp_nbs(), gdp_releases()["首次公布"]
    return rel.combine_first(nbs).sort_index()


def vintage(raw: pd.DataFrame, T: pd.Timestamp, q: pd.Period) -> pd.DataFrame:
    """月末 T 能看到的月度数据，行一直延伸到目标季度的季末（之后是缺口，由模型外推）。"""
    idx = pd.date_range(START, q.end_time.normalize() + pd.offsets.MonthEnd(0), freq="ME")
    x = raw[ACT].reindex(idx)
    for k in ACT:
        x.loc[available_at(k, x.index) > T, k] = np.nan
    x.loc[COVID_M[0]:COVID_M[1]] = np.nan
    return x[[k for k in ACT if x[k].notna().sum() >= MIN_OBS]]


def factor(x: pd.DataFrame) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.tsa.DynamicFactorMQ(x, factors=1, factor_orders=2, idiosyncratic_ar1=True).fit(disp=False, maxiter=500)
    f = res.factors.smoothed.iloc[:, 0]
    return f if f.corr(x["pmi"]) >= 0 else -f


def ols_predict(y: pd.Series, X: pd.DataFrame, x_new: pd.Series) -> float:
    r = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    return float(r.params["const"] + (r.params.drop("const") * x_new[r.params.drop("const").index]).sum())


def nowcast_one(raw: pd.DataFrame, gdp: pd.Series, T: pd.Timestamp, q: pd.Period) -> dict:
    x = vintage(raw, T, q)
    f = factor(x)
    fq = f.groupby(f.index.to_period("Q")).mean()
    known = gdp[gdp.index <= q - 1]
    lag = known.copy()
    lag.index = lag.index + 1                                            # 上季度的 GDP 放到本季度这一行
    train = [p for p in known.index if p not in COVID_Q and p in fq.index and p in lag.index]
    tr = pd.DataFrame({"y": known[train], "f": fq[train], "lag": lag[train]})
    out = {"上季度不变": float(known.iloc[-1])}
    new = pd.Series({"f": fq[q], "lag": float(known.iloc[-1])})
    out["因子模型"] = ols_predict(tr["y"], tr[["f", "lag"]], new)
    out["AR(1)"] = ols_predict(tr["y"], tr[["lag"]], new)
    ip = raw["ip_yoy"].copy()
    ip[available_at("ip_yoy", ip.index) > T] = np.nan
    ip.loc[COVID_M[0]:COVID_M[1]] = np.nan
    ipq = ip.groupby(ip.index.to_period("Q")).mean()
    ip_now = ipq.get(q, np.nan)
    if np.isnan(ip_now):
        ip_now = float(ip.dropna().iloc[-1])                              # 本季度还没有公布的月份：用最近一个月
    tr_ip = tr.assign(ip=ipq.reindex(tr.index)).dropna()
    out["工业增加值桥接"] = ols_predict(tr_ip["y"], tr_ip[["ip", "lag"]], pd.Series({"ip": ip_now, "lag": float(known.iloc[-1])}))
    out["因子（季度平均）"] = float(fq[q])
    out["指标数"] = x.shape[1]
    return out


def dm_test(e1: np.ndarray, e2: np.ndarray, lags: int = 1) -> tuple[float, float]:
    """Diebold-Mariano：损失差 = e1² − e2²，Newey-West 标准误。返回 (t, 双侧 p)。负的 t 表示第一个更准。"""
    d = e1 ** 2 - e2 ** 2
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.tvalues[0]), float(r.pvalues[0])


def main() -> None:
    raw = raw_panel()
    gdp, rel, nbs = gdp_target(), gdp_releases(), gdp_nbs()
    both = pd.concat([rel["首次公布"], nbs], axis=1, keys=["首次", "现在"]).dropna()
    diff = (both["现在"] - both["首次"]).abs()
    print(f"GDP 修订：{len(both)} 个季度，首次公布和现在的数相差的绝对值平均 {diff.mean():.2f}，最大 {diff.max():.1f}（{diff.idxmax()}），相同的 {(diff < 0.05).mean():.0%}")
    rows = []
    for q in pd.period_range(FIRST_Q, LIVE_Q, freq="Q"):
        if q in COVID_Q:
            continue
        for m in (1, 2, 3):
            T = (q.start_time + pd.offsets.MonthEnd(m)).normalize()
            if T > pd.Timestamp.today().normalize():
                continue
            row = {"季度": str(q), "第几个月": m, "月末": T.date(), **nowcast_one(raw, gdp, T, q)}
            row["首次公布"] = rel["首次公布"].get(q, np.nan)
            row["统计局现在的数"] = nbs.get(q, np.nan)
            row["一致预期"] = rel["一致预期"].get(q, np.nan)
            rows.append(row)
            print(row["季度"], m, {k: round(v, 2) for k, v in row.items() if isinstance(v, float)}, flush=True)
    fc = pd.DataFrame(rows)
    fc.to_csv(RESULTS / "nowcast_forecasts.csv", index=False, encoding="utf-8-sig")
    models = ["因子模型", "AR(1)", "上季度不变", "工业增加值桥接", "一致预期"]
    ev = fc[(fc["季度"].map(lambda s: pd.Period(s, "Q")) <= LAST_EVAL_Q)].dropna(subset=["首次公布"])
    scores = []
    for m in (1, 2, 3):
        g = ev[ev["第几个月"] == m]
        gc = g.dropna(subset=["一致预期"])
        for name in models:
            gg = gc if name == "一致预期" else g
            e = (gg[name] - gg["首次公布"]).to_numpy()
            row = {"第几个月": m, "模型": name, "季度数": len(e), "均方根误差": float(np.sqrt(np.mean(e ** 2))), "平均绝对误差": float(np.mean(np.abs(e)))}
            if name != "因子模型":
                e_f = (gg["因子模型"] - gg["首次公布"]).to_numpy()
                row["DM t（因子模型 − 它）"], row["DM p"] = dm_test(e_f, e)
            scores.append(row)
    sc = pd.DataFrame(scores)
    sc.to_csv(RESULTS / "nowcast_scores.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(sc.round(3).to_string(index=False))
    ext = fc[(fc["季度"].map(lambda s: pd.Period(s, "Q")) > LAST_EVAL_Q)]
    print("\n2025Q3 以后（对照统计局现在的数；2026Q3 还没公布）：")
    print(ext[["季度", "第几个月", "因子模型", "AR(1)", "上季度不变", "工业增加值桥接", "统计局现在的数"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
