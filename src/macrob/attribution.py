"""第 16 步 A、B：择时失败按环节归因、风险预算的来源归因。运行：uv run python -m macrob.attribution

规则（看结果之前定的，写在计划文件第 16 步里）：
- 16A 用第 14 步的实时 PCA 主方案（2017-02 至 2026-09）。事后状态 = 同一个模型用全样本估出的平滑概率 > 0.5。
  ① 识别：象限一致的比例、Brier 分数，对照随机猜；实时切换比事后切换晚几个月。
  ② 映射：按事后象限（当月）和实时象限（上月末，能交易的口径）分组，看各资产的月超额收益。
  ③ 定价：实时确认切换前后各 6 个月的累计超额收益（按方向折算）。
  ④ 执行：同一套择时信号，比较计不计成本、设不设换手上限。
  看完结果后加的（标明是修正）：全样本平滑概率 10 年里只切换 2 次，分的是疫情前后的波动大小，不像增长周期；
  所以映射另用和模型无关的参考周期（PMI、PPI 的 Bry-Boschan 拐点）定事后象限，再做一遍。
- 16B 用三类资产的风险平价、固定预算：收益贡献（债券再拆票息和利率变动）、风险贡献、放大到 60/40 的波动后比较、
  和固收+ 常用配比（80/20、90/10）及全部持有债券比较；前后两段分开做。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import allocation as A, cycles, regime, riskbudget as rb, sources as S, state
from .datalayer import RESULTS, raw_panel

QUAD = ["复苏", "过热", "滞胀", "衰退"]
ASSET4 = ["股", "债", "黄金", "商品"]
CLOCK_BEST = {"复苏": "股", "过热": "商品", "滞胀": "黄金", "衰退": "债"}
HALVES = (("前半段", "2017-02-01", "2021-09-30"), ("后半段", "2021-10-01", "2026-09-30"))


def _quadrant(g: pd.Series, i: pd.Series) -> pd.Series:
    return pd.Series(np.select([g & ~i, g & i, ~g & i], QUAD[:3], QUAD[3]), index=g.index)


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else np.nan


def reference_phase(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """参考周期：最近一个拐点是谷 → 上行（True），是峰 → 下行（False）。"""
    tp = cycles.turning_points(s.dropna())
    return pd.Series([bool(tp.loc[tp["date"] <= t, "kind"].iloc[-1] == "trough") if (tp["date"] <= t).any() else np.nan for t in index], index=index)


def inputs():
    levels, cash = A.daily_levels(), A.cash_monthly()
    q = pd.read_csv(RESULTS / "quadrant_probabilities.csv", index_col=0, parse_dates=True)
    q.index = q.index + pd.offsets.MonthEnd(0)
    months = pd.date_range(q.index.min(), min(q.index.max(), levels.index.max()), freq="ME")
    mo = levels.resample("ME").last().pct_change(fill_method=None)
    ex = mo.sub(cash.reindex(mo.index), axis=0)
    return levels, cash, q, months, mo, ex


def part_a(q, months, ex, levels, cash) -> dict[str, pd.DataFrame]:
    dims = state.dimensions(state.category_factors())
    post = {}
    for k, col in (("增长", "增长加速"), ("通胀", "通胀加速")):
        y = (dims[k] - dims[k].shift(3)).dropna()
        post[col] = regime.smoothed(y)
    post = pd.DataFrame(post).reindex(q.index)
    g_rt, i_rt = q["增长加速"] > 0.5, q["通胀加速"] > 0.5
    g_po, i_po = post["增长加速"] > 0.5, post["通胀加速"] > 0.5
    quad_rt, quad_po = _quadrant(g_rt, i_rt), _quadrant(g_po, i_po)
    freq = quad_po.value_counts(normalize=True)
    # ① 识别
    lags, matched = [], 0
    sw_po = regime.switches(post["增长加速"].dropna())
    sw_rt = regime.switches(q["增长加速"].dropna(), confirm=True)
    for _, e in sw_po.iterrows():
        c = sw_rt[(sw_rt["kind"] == e["kind"]) & (sw_rt["date"] >= e["date"] - pd.DateOffset(months=6)) & (sw_rt["date"] <= e["date"] + pd.DateOffset(months=12))]
        if len(c):
            matched += 1
            d = c["date"].iloc[0]
            lags.append((d.year - e["date"].year) * 12 + d.month - e["date"].month)
    ident = pd.DataFrame([
        {"指标": "象限一致的月份比例", "实时": float((quad_rt == quad_po).mean()), "对照": "随机猜 0.25；按事后频率猜 " + f"{float((freq ** 2).sum()):.2f}"},
        {"指标": "增长方向一致", "实时": float((g_rt == g_po).mean()), "对照": "随机猜 0.50"},
        {"指标": "通胀方向一致", "实时": float((i_rt == i_po).mean()), "对照": "随机猜 0.50"},
        {"指标": "增长概率的 Brier 分数", "实时": float(((q["增长加速"] - g_po.astype(float)) ** 2).mean()), "对照": "一直猜 0.5 = 0.25"},
        {"指标": "通胀概率的 Brier 分数", "实时": float(((q["通胀加速"] - i_po.astype(float)) ** 2).mean()), "对照": "一直猜 0.5 = 0.25"},
        {"指标": "事后增长切换次数 / 实时对上的次数", "实时": f"{len(sw_po)} / {matched}", "对照": ""},
        {"指标": "实时确认比事后切换晚（月，平均）", "实时": float(np.mean(lags)) if lags else np.nan, "对照": ""},
    ])
    # ② 映射：事后象限用当月收益；实时象限用上月末的判断、当月收益
    rows = []
    g_ref, i_ref = reference_phase(cycles.pmi(), q.index), reference_phase(raw_panel()["ppi_yoy"], q.index)
    quad_ref = _quadrant(g_ref.astype(bool), i_ref.astype(bool))
    for label, quad, shift in (("事后象限（当月）", quad_po, 0), ("实时象限（上月末判断）", quad_rt, 1),
                               ("参考周期象限（当月，看完结果后加的）", quad_ref, 0)):
        qq = quad.shift(shift).reindex(ex.index)
        for k in QUAD:
            m = qq == k
            row = {"分组": label, "象限": k, "月数": int(m.sum()), "时钟预期最好的": CLOCK_BEST[k]}
            for a in ASSET4:
                row[f"{a}（月均超额 %）"] = float(ex.loc[m, a].mean() * 100)
                row[f"{a} t"] = _t(ex.loc[m, a])
            means = {a: row[f"{a}（月均超额 %）"] for a in ASSET4}
            row["实际最好的"] = max(means, key=means.get) if m.sum() else ""
            rows.append(row)
    mapping = pd.DataFrame(rows)
    # ③ 定价：实时确认切换前后各 6 个月（按方向折算：转入加速记 +1，转入减速记 −1）
    rows = []
    for label, sw in (("实时确认的切换", sw_rt), ("事后的切换", sw_po)):
        for _, e in sw.iterrows():
            d = 1 if e["kind"] == "trough" else -1
            pos = ex.index.searchsorted(e["date"])
            if pos - 5 < 0 or pos + 6 >= len(ex):
                continue
            rows.append({"口径": label, "日期": e["date"], "方向": d,
                         **{f"{a} 之前 6 个月": float(ex[a].iloc[pos - 5: pos + 1].sum() * d * 100) for a in ("股", "债")},
                         **{f"{a} 之后 6 个月": float(ex[a].iloc[pos + 1: pos + 7].sum() * d * 100) for a in ("股", "债")}})
    pricing = pd.DataFrame(rows)
    summ = []
    for label, g in pricing.groupby("口径", sort=False):
        summ.append({"口径": label, "次数": len(g), **{f"{c}（均值）": float(g[c].mean()) for c in pricing.columns[3:]},
                     **{f"{c} t": _t(g[c]) for c in pricing.columns[3:]}})
    pricing = pd.concat([pricing, pd.DataFrame(summ)], ignore_index=True)
    # ④ 执行
    prob = A.budgets(q, False)
    rows = []
    for name, cost, cap in (("不计成本、不设上限", np.zeros(3), None), ("不计成本、设上限", np.zeros(3), A.MAX_TO),
                            ("计成本、不设上限", A.COST, None), ("计成本、设上限（实际用的）", A.COST, A.MAX_TO)):
        r, W = A.run(prob, levels, months, max_to=cap, cost=cost)
        rows.append({"情况": name, **A.metrics(r, cash, W)})
    exe = pd.DataFrame(rows)
    return {"识别": ident, "映射": mapping, "定价": pricing, "执行": exe}


def part_b(q, months, mo, levels, cash) -> dict[str, pd.DataFrame]:
    fixed_fn = lambda t: np.mean([A.budgets(q.loc[:t], False)(d) for d in q.loc[:t].index], axis=0)
    strategies = {"风险平价": A.run(lambda t: np.ones(3) / 3, levels, months, max_to=None),
                  "固定预算（不择时）": A.run(fixed_fn, levels, months, max_to=None)}
    mixes = {"60/40（股 / 债）": [0.6, 0.0, 0.4], "固收+ 80/20（债 / 股）": [0.2, 0.0, 0.8], "固收+ 90/10（债 / 股）": [0.1, 0.0, 0.9], "全部持有债券": [0.0, 0.0, 1.0]}
    others = {k: A.run(None, levels, months, fixed_w=w) for k, w in mixes.items()}
    start = max(r.index.min() for r, _ in strategies.values())
    y10 = S.yields_daily()["y10"].resample("ME").last()
    carry = (y10.shift(1) / 1200).reindex(mo.index)                      # 票息：上月末 10 年收益率的月化
    wk = levels.resample("W-FRI").last().pct_change(fill_method=None)
    contrib, risk, comp = [], [], []
    for period, a, b in (("全样本", "2017-02-01", "2026-09-30"),) + HALVES:
        for name, (r, W) in strategies.items():
            W2 = W.copy()
            W2.index = [months[months.get_loc(t) + 1] for t in W2.index]       # 决策月 → 收益月
            W2 = W2[(W2.index >= max(pd.Timestamp(a), start)) & (W2.index <= b)]
            exr = mo.loc[W2.index, A.ASSETS].sub(cash.reindex(W2.index), axis=0)
            c = (W2 * exr).sum() * 12 / len(W2) * 100                          # 年化的超额收益贡献（百分点）
            bond_carry = float((W2["债"] * (carry.reindex(W2.index) - cash.reindex(W2.index))).sum() * 12 / len(W2) * 100)
            contrib.append({"区间": period, "方案": name, "股": c["股"], "黄金": c["黄金"], "债·票息减现金": bond_carry,
                            "债·利率变动等": c["债"] - bond_carry, "合计": c.sum()})
            rc = []
            for t in W.index:
                if not (pd.Timestamp(a) <= months[months.get_loc(t) + 1] <= pd.Timestamp(b)):
                    continue
                win = wk.loc[:t, A.ASSETS].dropna().iloc[-A.WIN_W:]
                Sm = rb.ledoit_wolf_corr(win.to_numpy())[0] * 52 / 12
                rc.append(rb.risk_contributions(W.loc[t].to_numpy(), Sm))
            risk.append({"区间": period, "方案": name, **dict(zip(A.ASSETS, np.mean(rc, axis=0)))})
        r6040 = others["60/40（股 / 债）"][0]
        r6040 = r6040[(r6040.index >= max(pd.Timestamp(a), start)) & (r6040.index <= b)]
        vol6040 = r6040.std()
        for name, (r, W) in {**strategies, **others}.items():
            r = r[(r.index >= max(pd.Timestamp(a), start)) & (r.index <= b)]
            comp.append({"区间": period, "方案": name, "口径": "原始", **A.metrics(r, cash)})
            if name in strategies:
                rf = cash.reindex(r.index).fillna(0)
                scaled = rf + (r - rf) * vol6040 / r.std()
                comp.append({"区间": period, "方案": name, "口径": "放大到 60/40 的波动", "放大倍数": float(vol6040 / r.std()), **A.metrics(scaled, cash)})
    return {"收益贡献": pd.DataFrame(contrib), "风险贡献": pd.DataFrame(risk), "比较": pd.DataFrame(comp)}


def main() -> None:
    levels, cash, q, months, mo, ex = inputs()
    a = part_a(q, months, ex, levels, cash)
    b = part_b(q, months, mo, levels, cash)
    pd.set_option("display.width", 250)
    for name, t in {**{f"16A·{k}": v for k, v in a.items()}, **{f"16B·{k}": v for k, v in b.items()}}.items():
        t.to_csv(RESULTS / f"attribution_{name.replace('·', '_')}.csv", index=False, encoding="utf-8-sig")
        print(f"\n===== {name}")
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
