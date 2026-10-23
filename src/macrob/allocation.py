"""第 4、5 步：可投资资产 + 概率美林时钟的风险预算配置。运行：uv run python -m macrob.allocation

资产：沪深300 全收益指数；债券 = 中债国债财富指数（7–10 年，第 12 步补到的，计划原本就写了「能找到就用指数」），
用收益率合成的 10 年国债总收益留作对照（债(合成)）；黄金 ETF（518880）；
商品 = 中证商品期货综合指数（H11061，第 12 步补到的），按期货价格计算、假定不含保证金利息，加上 3 个月 SHIBOR；
现金按 3 个月 SHIBOR 计息。

规则（看结果之前定的，和实习补做时同一套映射）：
- 风险预算：股 = 增长加速 ×（1 − 通胀加速），黄金 = 增长加速 × 通胀加速，债 = 1 − 增长加速；每类至少 2%。
- 协方差：过去 104 周的周收益（至少 52 周），对相关系数做 Ledoit-Wolf 收缩。
- 每月末定目标权重，下个月生效；月度双边换手上限 3%；单边成本 股 0.15%、黄金 0.10%、债 0.05%。
- 对照：60/40（股 / 债）、风险平价（预算各 1/3）、固定预算（到当时为止的历史平均概率预算，不择时）。
- 评价：扣 3 个月 SHIBOR 后的夏普、年化、波动、最大回撤、卡尔马；平移安慰剂 300 次检验择时部分。
- 第 12 步加的四类资产版本（经典美林时钟）：股 = 复苏，商品 = 过热，黄金 = 滞胀，债 = 衰退；商品单边成本 0.10%。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import riskbudget as rb, sources as S
from .datalayer import RESULTS
from .probe_util import RAW

ASSETS = ["股", "黄金", "债"]
COST = np.array([0.0015, 0.0010, 0.0005])
ASSETS4 = ["股", "商品", "黄金", "债"]
COST4 = np.array([0.0015, 0.0010, 0.0010, 0.0005])
WIN_W, MIN_W, MAX_TO, FLOOR, N_PERM = 104, 52, 0.03, 0.02, 300
TR = S.EXTERNAL / "total_return_index.parquet"


def daily_levels() -> pd.DataFrame:
    t = pd.read_parquet(TR)
    stock = t[t["code"] == "H00300"].set_index("date")["close"].astype(float)
    stock.index = pd.to_datetime(stock.index)
    y = S.yields_daily()["y10"].dropna() / 100
    dy = y.diff()
    r = y.shift(1) / 252 - 8 * dy + 0.5 * 80 * dy ** 2
    synth = (1 + r.fillna(0)).cumprod()
    cb = pd.read_csv(RAW / "cbond_treasury_wealth_7-10Y.csv", parse_dates=["date"]).set_index("date")["value"].astype(float)
    gold = pd.read_csv(RAW / "etf_518880_sina.csv", parse_dates=["date"]).set_index("date")["close"].astype(float)   # 和实习补做时的文件逐日一致，补到最新
    cm = pd.read_csv(RAW / "csindex_H11061.csv", parse_dates=["日期"]).sort_values("日期").set_index("日期")["收盘"].astype(float)
    cm = cm[cm.index >= "2005-01-01"]
    shibor = S.shibor3m_daily().reindex(cm.index).ffill() / 100 / 252
    comm = (1 + (cm.pct_change() + shibor).fillna(0)).cumprod()            # 期货指数 + 保证金收益
    return pd.concat({"股": stock, "商品": comm, "黄金": gold, "债": cb, "债(合成)": synth}, axis=1).sort_index()


def cash_monthly() -> pd.Series:
    return S.shibor3m_daily().resample("ME").mean() / 1200


def floor_budgets(b, floor=FLOOR):
    b = np.maximum(np.asarray(b, float), 0)
    b = b / b.sum()
    return (1 - floor * len(b)) * b + floor


def run(budget_fn, levels: pd.DataFrame, months: pd.DatetimeIndex, max_to=MAX_TO, fixed_w=None,
        assets=ASSETS, cost=COST) -> tuple[pd.Series, pd.DataFrame]:
    """budget_fn(t) → 风险预算；fixed_w 给定时直接用固定权重（60/40）。返回 (月度净收益, 权重)。"""
    wk = levels.resample("W-FRI").last().pct_change(fill_method=None)
    mo = levels.resample("ME").last().pct_change(fill_method=None)
    rets, W, prev = {}, {}, None
    for t in months[:-1]:
        nxt = months[months.get_loc(t) + 1]
        if fixed_w is not None:
            w = np.asarray(fixed_w, float)
        else:
            win = wk.loc[:t, assets].dropna().iloc[-WIN_W:]
            if len(win) < MIN_W:
                continue
            Sm = rb.ledoit_wolf_corr(win.to_numpy())[0] * 52 / 12
            b = floor_budgets(budget_fn(t))
            target = rb.risk_budget_weights(Sm, b)
            w = rb.constrained_rebalance(target, prev, Sm, max_to) if (prev is not None and max_to) else target
        r_next = mo.loc[nxt, assets].to_numpy()
        if np.isnan(r_next).any():
            continue
        c = float(np.abs(w - (prev if prev is not None else w)) @ cost)
        rets[nxt] = float(w @ r_next) - c
        W[t] = w
        prev = w
    return pd.Series(rets).sort_index(), pd.DataFrame(W, index=assets).T


def metrics(r: pd.Series, cash: pd.Series, W: pd.DataFrame | None = None) -> dict:
    ex = r - cash.reindex(r.index).fillna(0)
    nav = (1 + r).cumprod()
    ann = float(nav.iloc[-1] ** (12 / len(r)) - 1)
    mdd = float((nav / nav.cummax() - 1).min())
    out = {"月数": len(r), "年化": ann, "波动": float(r.std() * np.sqrt(12)), "扣无风险夏普": float(ex.mean() / ex.std() * np.sqrt(12)),
           "最大回撤": mdd, "卡尔马": ann / abs(mdd) if mdd < 0 else np.nan}
    if W is not None and len(W):
        out["月均双边换手"] = float(W.diff().abs().sum(axis=1).iloc[1:].mean())
        out.update({f"平均权重·{a}": float(W[a].mean()) for a in W.columns})
    return out


def budgets(q: pd.DataFrame, four: bool):
    """象限概率 → 风险预算。三类：股 = 复苏，黄金 = 过热，债 = 滞胀 + 衰退；四类：股、商品、黄金、债 = 复苏、过热、滞胀、衰退。"""
    def f(t, q=q):
        g, i = q.loc[t, "增长加速"], q.loc[t, "通胀加速"]
        return np.array([g * (1 - i), g * i, (1 - g) * i, (1 - g) * (1 - i)]) if four else np.array([g * (1 - i), g * i, 1 - g])
    return f


def study(levels, cash, q, months, four: bool) -> tuple[pd.DataFrame, np.ndarray, dict]:
    assets, cost = (ASSETS4, COST4) if four else (ASSETS, COST)
    n_a = len(assets)
    prob = budgets(q, four)
    def fixed(t):
        h = q.loc[:t]
        return np.mean([budgets(h, four)(d) for d in h.index], axis=0)
    w6040 = [0.6, 0.0, 0.0, 0.4] if four else [0.6, 0.0, 0.4]
    kw = dict(assets=assets, cost=cost)
    res = {
        "60/40（股 / 债）": run(None, levels, months, fixed_w=w6040, **kw),
        "风险平价": run(lambda t: np.ones(n_a) / n_a, levels, months, max_to=None, **kw),
        "固定预算（不择时）": run(fixed, levels, months, max_to=None, **kw),
        "概率预算（择时）": run(prob, levels, months, **kw),
        "概率预算（不设换手上限）": run(prob, levels, months, max_to=None, **kw),
    }
    start = max(r.index.min() for r, _ in res.values())
    rows = []
    for name, (r, W) in res.items():
        r = r[r.index >= start]
        rows.append({"方案": name, **metrics(r, cash, W.loc[W.index >= start - pd.offsets.MonthEnd(1)] if len(W) else None)})
    t = pd.DataFrame(rows)
    # 平移安慰剂：把象限概率整体错开 12 个月以上
    rng = np.random.default_rng(0)
    fake = []
    for k in rng.integers(12, len(q) - 12, size=N_PERM):
        qs = pd.DataFrame(np.roll(q[["增长加速", "通胀加速"]].to_numpy(), k, axis=0), index=q.index, columns=["增长加速", "通胀加速"])
        r, _ = run(budgets(qs, four), levels, months, **kw)
        fake.append(metrics(r[r.index >= start], cash)["扣无风险夏普"])
    return t, np.array(fake), res


def main() -> None:
    levels = daily_levels()
    cash = cash_monthly()
    # 第 14 步：主方案是 PCA 扩张窗口的概率；等权、PCA 5 年滚动窗口作对照，三个版本用同一段月份
    probs = {v: pd.read_csv(RESULTS / f"quadrant_probabilities{v}.csv", index_col=0, parse_dates=True) for v in ("", "_等权", "_滚动")}
    for q in probs.values():
        q.index = q.index + pd.offsets.MonthEnd(0)
    months = pd.date_range(max(q.index.min() for q in probs.values()),
                           min(min(q.index.max() for q in probs.values()), levels.index.max()), freq="ME")
    pd.set_option("display.width", 260)
    for v, q in probs.items():
        for four, suffix in ((False, ""), (True, "_4assets")):
            t, fake, res = study(levels, cash, q, months, four)
            real = t.loc[t["方案"] == "概率预算（择时）", "扣无风险夏普"].iloc[0]
            t.to_csv(RESULTS / f"allocation_summary{v}{suffix}.csv", index=False, encoding="utf-8-sig")
            pd.Series(fake, name="平移后的扣无风险夏普").to_csv(RESULTS / f"allocation_placebo{v}{suffix}.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({k: v_[0] for k, v_ in res.items()}).to_csv(RESULTS / f"allocation_returns{v}{suffix}.csv", encoding="utf-8-sig")
            print({"": "主方案·PCA", "_等权": "对照·等权", "_滚动": "对照·PCA 5 年滚动"}[v], "·", "四类资产（加商品）" if four else "三类资产",
                  f"；换手上限被放开 {rb.RELAXED['count']} 次")
            print(t.round(4).to_string(index=False))
            print(f"平移安慰剂（{N_PERM} 次）：夏普均值 {fake.mean():.3f}，不差于真实择时的概率 {(fake >= real).mean():.3f}\n")

if __name__ == "__main__":
    main()
