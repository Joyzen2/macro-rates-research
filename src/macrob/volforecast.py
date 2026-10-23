"""第 16 步 C：在市场自身的波动、相关性和期权隐含波动率之外，宏观变量能否改善固收+ 组合下月波动率的预测。
运行：uv run python -m macrob.volforecast

规则（看结果之前定的，写在计划文件第 16 步里）：
- 组合：债 80%（中债 7–10 年国债财富指数）+ 股 20%（沪深300 全收益），每月初再平衡；稳健性 90/10、70/30。
- 预测对象：下个月的实现波动率（当月日收益算、年化），取对数。
- 市场基准：组合过去 1、3、12 个月的实现波动（HAR），股、债各自当月的波动，股债当月日收益的相关。
  比较二再加 50ETF 期权 30 天平值隐含波动率（月末，取对数），2015-02 起。
- 宏观（月末当时能看到的）：实时 PCA 增长、通胀因子 3 个月变化的绝对值；当月数据意外绝对值的均值（截到 3）；
  当月有没有央行公告、利率互换意外绝对值之和。主检验一起加；另报分组加；社融信贷脉冲单独加（2014 起）。
- 扩张窗口 OLS，每月重估；比较一至少 60 个月，比较二至少 36 个月；两个模型用同一批训练月份和评估月份。
  损失：对数波动的平方误差、QLIKE；嵌套模型用 Clark-West，QLIKE 差用 Diebold-Mariano。
- 经济价值：股票仓位 = 基准仓位 × 目标波动 / 预测波动，限制在 0 到 1.5 倍基准仓位（80/20 时是 0–30%）；
  目标 = 到当月为止组合实现波动的均值；和固定配比比较，计成本。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import allocation as A, credit, sources as S, state
from .datalayer import RESULTS, ROOT

OPT = ROOT / "data" / "manual" / "csmar_options_20261022"
ROUND1_EVENTS = S.UPGRADE_DATA.parent / "results" / "surprise_events.csv"
MIXES = {"80/20": 0.2, "90/10": 0.1, "70/30": 0.3}                  # 股票权重
BASE = ["组合波动_1月", "组合波动_3月", "组合波动_12月", "股波动", "债波动", "股债相关"]
MACRO = {"增长通胀因子变化": ["增长变化", "通胀变化"], "数据意外": ["数据意外"], "政策公告": ["有政策公告", "利率互换意外"]}
COST = {"股": 0.0015, "债": 0.0005}


def month_end(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)


def daily_returns() -> pd.DataFrame:
    return A.daily_levels()[["股", "债"]].dropna().pct_change().dropna()


def portfolio_daily(r: pd.DataFrame, w_stock: float) -> pd.Series:
    """每月初回到目标权重，月内权重随价格漂移。"""
    out, w, cur = {}, None, None
    for t, x in zip(r.index, r[["股", "债"]].to_numpy()):
        m = (t.year, t.month)
        if m != cur:
            w, cur = np.array([w_stock, 1 - w_stock]), m
        out[t] = float(w @ x)
        w = w * (1 + x)
        w = w / w.sum()
    return pd.Series(out)


def monthly_vol(x: pd.Series) -> pd.Series:
    v = x.groupby(month_end(x.index)).apply(lambda s: float(np.sqrt(252 * np.mean(s.to_numpy() ** 2))))
    return v


def implied_vol_30d(underlying: str = "510050") -> pd.Series:
    """日度 30 天平值隐含波动率：最近两个到期月（剩余至少 7 天）各取最接近平值的看涨、看跌求平均，按 VIX 的做法插值方差。"""
    d = pd.read_parquet(OPT / "stock_option_iv.parquet")
    d = d[(d["标的"] == underlying) & (d["ImpliedVolatility"] > 0)].copy()
    d["天数"] = (d["ExerciseDate"] - d["TradingDate"]).dt.days
    d = d[d["天数"] >= 7]
    out = {}
    for t, g in d.groupby("TradingDate"):
        s0 = g["UnderlyingScrtClose"].median()
        terms = []
        for _, h in g.groupby("ExerciseDate"):
            k = h["StrikePrice"].iloc[(h["StrikePrice"] - s0).abs().argmin()]
            terms.append((int(h["天数"].iloc[0]), float(h.loc[h["StrikePrice"] == k, "ImpliedVolatility"].mean())))
        terms.sort()
        out[t] = interp_30d(terms)
    return pd.Series(out).sort_index()


def interp_30d(terms: list[tuple[int, float]], target: int = 30) -> float:
    """terms：按剩余天数排好的 (天数, 隐含波动率)。"""
    if len(terms) == 1 or terms[0][0] >= target:
        return terms[0][1]
    (t1, s1), (t2, s2) = terms[0], terms[1]
    if t2 <= target:
        return s2
    var = (t1 * s1 ** 2 * (t2 - target) / (t2 - t1) + t2 * s2 ** 2 * (target - t1) / (t2 - t1)) / target
    return float(np.sqrt(var))


def macro_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    dims = state.dimensions(state.category_factors())
    f = pd.DataFrame(index=index)
    f["增长变化"] = (dims["增长"] - dims["增长"].shift(3)).abs().reindex(index)
    f["通胀变化"] = (dims["通胀"] - dims["通胀"].shift(3)).abs().reindex(index)
    r1 = pd.read_csv(ROUND1_EVENTS, parse_dates=["date"])
    r1 = r1[r1["indicator"] != "CPI"][["date", "surprise"]]
    cpi = pd.read_csv(RESULTS / "surprise_cpi_延长_events.csv", parse_dates=["date"])[["date", "surprise"]]
    nbs = pd.read_csv(RESULTS / "surprise_nbs_events_延长.csv", parse_dates=["时间"])
    nbs = nbs[nbs["指标"].isin(["社零", "固投"])].assign(date=lambda x: x["时间"].dt.normalize())[["date", "surprise"]]
    ev = pd.concat([r1, cpi, nbs], ignore_index=True).dropna()
    first = ev["date"].min()
    sup = ev.assign(a=ev["surprise"].abs().clip(upper=3)).groupby(month_end(ev["date"]))["a"].mean()
    f["数据意外"] = sup.reindex(index)
    f.loc[f.index < month_end([first])[0], "数据意外"] = np.nan
    pe = pd.read_csv(RESULTS / "policy_irs_events.csv", parse_dates=["date"])
    pm = month_end(pe["date"])
    f["有政策公告"] = pd.Series(1.0, index=pm).groupby(level=0).max().reindex(index).fillna(0.0)
    f["利率互换意外"] = pe["Δ互换1Y(bp)"].abs().groupby(pm).sum().reindex(index).fillna(0.0)
    f.loc[f.index < pd.Timestamp("2010-01-31"), ["有政策公告", "利率互换意外"]] = np.nan
    f["信贷脉冲"] = credit.tsf_impulse()["月末可用的信贷脉冲"].reindex(index)
    return f


def panel(w_stock: float, iv: pd.Series | None) -> pd.DataFrame:
    r = daily_returns()
    p = portfolio_daily(r, w_stock)
    rv = monthly_vol(p)
    idx = rv.index
    d = pd.DataFrame(index=idx)
    d["组合波动_1月"] = np.log(rv)
    d["组合波动_3月"] = np.log(rv.rolling(3).mean())
    d["组合波动_12月"] = np.log(rv.rolling(12).mean())
    d["股波动"] = np.log(monthly_vol(r["股"]).reindex(idx))
    d["债波动"] = np.log(monthly_vol(r["债"]).reindex(idx))
    d["股债相关"] = r.groupby(month_end(r.index)).apply(lambda g: g["股"].corr(g["债"])).reindex(idx)
    if iv is not None:
        d["隐含波动"] = np.log(iv.groupby(month_end(iv.index)).last().reindex(idx))
    d = d.join(macro_features(idx))
    d["y"] = d["组合波动_1月"].shift(-1)                                 # 下个月的对数实现波动
    return d


def oos(d: pd.DataFrame, cols: list[str], need: list[str], min_n: int) -> pd.DataFrame:
    """月末 i 只用 y 已经实现的月份（j < i）估计，预测 i 的下一个月。need = 两个模型都要有值的列。"""
    rows = []
    for i in range(len(d)):
        train = d.iloc[:i].dropna(subset=need + ["y"])
        x = d.iloc[i]
        if len(train) < min_n or x[need].isna().any() or pd.isna(x["y"]):
            continue
        fit = sm.OLS(train["y"], sm.add_constant(train[cols])).fit()
        rows.append({"月份": d.index[i], "预测": float(fit.params["const"] + (fit.params[cols] * x[cols]).sum()),
                     "残差方差": float(fit.mse_resid), "实际": float(x["y"])})
    return pd.DataFrame(rows).set_index("月份")


def qlike(true_log_vol: pd.Series, pred: pd.Series, s2: pd.Series) -> pd.Series:
    ratio = np.exp(2 * true_log_vol) / np.exp(2 * pred + 2 * s2)          # 对数正态修正
    return ratio - np.log(ratio) - 1


def compare(d: pd.DataFrame, base: list[str], add: list[str], min_n: int, name: str) -> dict:
    need = base + add
    a, b = oos(d, base, need, min_n), oos(d, base + add, need, min_n)
    e1, e2 = a["实际"] - a["预测"], b["实际"] - b["预测"]
    f = e1 ** 2 - e2 ** 2 + (a["预测"] - b["预测"]) ** 2                  # Clark-West
    cw = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    q1, q2 = qlike(a["实际"], a["预测"], a["残差方差"]), qlike(b["实际"], b["预测"], b["残差方差"])
    dm = sm.OLS(q1 - q2, np.ones(len(q1))).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return {"比较": name, "样本外": f"{a.index.min():%Y-%m} ~ {a.index.max():%Y-%m}", "月数": len(a),
            "基准 MSE": float((e1 ** 2).mean()), "加宏观 MSE": float((e2 ** 2).mean()), "样本外 R²": float(1 - (e2 ** 2).mean() / (e1 ** 2).mean()),
            "Clark-West t": float(cw.tvalues.iloc[0]), "基准 QLIKE": float(q1.mean()), "加宏观 QLIKE": float(q2.mean()),
            "QLIKE 差的 t": float(dm.tvalues.iloc[0]), "_forecasts": (a, b)}


def vol_target(d: pd.DataFrame, fc: pd.DataFrame, w0: float, cash: pd.Series) -> dict:
    """月末 m 用对下月的波动预测定股票仓位；目标 = 到 m 为止组合实现波动的均值。"""
    lv = A.daily_levels()[["股", "债"]].resample("ME").last().pct_change(fill_method=None)
    rv = np.exp(d["组合波动_1月"])
    rets, prev, ws = {}, None, {}
    for m, row in fc.iterrows():
        nxt = m + pd.offsets.MonthEnd(1)
        if nxt not in lv.index:
            continue
        vol_hat = float(np.exp(row["预测"] + row["残差方差"] / 2))
        w = float(np.clip(w0 * rv.loc[:m].mean() / vol_hat, 0, 1.5 * w0))
        c = 0.0 if prev is None else abs(w - prev) * (COST["股"] + COST["债"])
        rets[nxt] = w * lv.loc[nxt, "股"] + (1 - w) * lv.loc[nxt, "债"] - c
        ws[m], prev = w, w
    r = pd.Series(rets)
    fixed = pd.Series({m + pd.offsets.MonthEnd(1): w0 * lv.loc[m + pd.offsets.MonthEnd(1), "股"] + (1 - w0) * lv.loc[m + pd.offsets.MonthEnd(1), "债"]
                       for m in fc.index if m + pd.offsets.MonthEnd(1) in lv.index})
    out = {}
    for label, x in (("波动目标", r), ("固定配比", fixed)):
        mt = A.metrics(x, cash)
        out[label] = {k: mt[k] for k in ("月数", "年化", "波动", "扣无风险夏普", "最大回撤")}
    out["波动目标"]["平均股票仓位"] = float(np.mean(list(ws.values())))
    out["波动目标"]["月均仓位变动"] = float(pd.Series(ws).diff().abs().mean())
    return out


def main() -> None:
    iv = implied_vol_30d()
    cash = A.cash_monthly()
    pd.set_option("display.width", 250)
    rows, econ = [], []
    for mix, w0 in MIXES.items():
        d = panel(w0, iv)
        macro_all = sum(MACRO.values(), [])
        specs = [("比较一：市场基准 + 宏观（三项一起）", BASE, macro_all, 60), ("比较二：市场基准 + 隐含波动 + 宏观", BASE + ["隐含波动"], macro_all, 36)]
        if mix == "80/20":
            specs += [(f"比较一：只加{k}", BASE, v, 60) for k, v in MACRO.items()]
            specs += [("比较一：只加信贷脉冲（2014 起）", BASE, ["信贷脉冲"], 60), ("隐含波动本身：市场基准 + 隐含波动", BASE, ["隐含波动"], 36)]
        for name, base, add, min_n in specs:
            res = compare(d, base, add, min_n, name)
            a, b = res.pop("_forecasts")
            rows.append({"组合": mix, **res})
            if name.startswith(("比较一：市场基准", "比较二")):
                for label, fc in (("基准", a), ("加宏观", b)):
                    for k, v in vol_target(d, fc, w0, cash).items():
                        econ.append({"组合": mix, "比较": name[:3], "预测用的模型": label if k == "波动目标" else "—", "方案": k, **v})
        if mix == "80/20":
            d.to_csv(RESULTS / "volforecast_panel_80_20.csv", encoding="utf-8-sig")
    t, e = pd.DataFrame(rows), pd.DataFrame(econ).drop_duplicates(subset=["组合", "比较", "方案", "预测用的模型", "月数"])
    t.to_csv(RESULTS / "volforecast_tests.csv", index=False, encoding="utf-8-sig")
    e.to_csv(RESULTS / "volforecast_vol_target.csv", index=False, encoding="utf-8-sig")
    iv.to_csv(RESULTS / "implied_vol_50etf_30d.csv", encoding="utf-8-sig")
    print(f"50ETF 30 天隐含波动：{iv.index.min():%Y-%m-%d} ~ {iv.index.max():%Y-%m-%d}，{len(iv)} 天，均值 {iv.mean():.3f}")
    print(t.round(4).to_string(index=False))
    print(e.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
