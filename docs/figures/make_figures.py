"""画网站上的图。运行：uv run python docs/figures/make_figures.py

读 results/ 里的结果表和 data/raw/ 里的中债曲线，图写到 docs/figures/svg/。
其中散点图要用每次公告的利率互换变化、状态概率图要重跑模型，都需要本地数据，仓库里没有。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import charts as C  # noqa: E402

PROJECT = pathlib.Path(__file__).resolve().parents[2]
R, RAW = PROJECT / "results", PROJECT / "data" / "raw"
OUT = pathlib.Path(__file__).parent / "svg"


def save(name: str, svg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
    print("写了", name)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv", encoding="utf-8-sig")


def fig_chain() -> str:
    boxes = [("宏观数据", "按首次公布日对齐", False), ("宏观状态", "增长、通胀的实时概率", False),
             ("市场预期", "一致预期、互换、期权", True), ("资产配置", "风险预算、固收+", False)]
    svg = C.Svg(118, "研究框架：宏观状态和资产配置之间，隔着市场预期这一环",
                "四个方框从左到右：宏观数据、宏观状态、市场预期、资产配置。市场预期这一环被强调。")
    gap, w, h, y = 12, (C.W - 3 * 12) / 4, 58, 8
    for i, (head, sub, strong) in enumerate(boxes):
        x = i * (w + gap)
        fill, stroke = ("#e8f1fc", C.S[0]) if strong else (C.SURFACE, C.BASE)
        svg.add(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{1.5 if strong else 1}"/>')
        svg.text(x + w / 2, y + 22, head, size=C.LABEL, anchor="middle", weight="600")
        svg.text(x + w / 2, y + 41, sub, size=10, cls="t2", anchor="middle")
        if i < 3:
            ax = x + w + 1
            svg.add(f'<path d="M{ax:.1f},{y + h / 2:.1f}h{gap - 4:.1f}" stroke="{C.MUTED}" stroke-width="1.4"/>'
                    f'<path d="M{ax + gap - 6:.1f},{y + h / 2 - 3.5:.1f}l4,3.5l-4,3.5" fill="none" stroke="{C.MUTED}" stroke-width="1.4"/>')
    x2, x4 = (w + gap) * 1 + w / 2, (w + gap) * 3 + w / 2
    svg.add(f'<path d="M{x2:.1f},{y + h + 4}V{y + h + 20}H{x4:.1f}V{y + h + 6}" fill="none" stroke="{C.BASE}" stroke-width="1.2"/>'
            f'<path d="M{x4 - 3.5:.1f},{y + h + 10}l3.5,-4l3.5,4" fill="none" stroke="{C.BASE}" stroke-width="1.2"/>')
    svg.text(C.W / 2, y + h + 34, "美林时钟从「宏观状态」直接跳到「资产配置」，省掉了中间这一环", size=11, cls="t3", anchor="middle")
    return svg.render()


def fig_identify() -> str:
    sys.path.insert(0, str(PROJECT / "src"))
    from macrob import regime, state
    q = pd.read_csv(R / "quadrant_probabilities.csv", index_col=0, parse_dates=True)
    q.index = q.index + pd.offsets.MonthEnd(0)
    dims = state.dimensions(state.category_factors())
    y = (dims["增长"] - dims["增长"].shift(3)).dropna()
    post = regime.smoothed(y).reindex(q.index)
    q = q.loc[:"2026-09-30"]
    post = post.reindex(q.index)
    dates = list(q.index)
    return C.line_chart(dates, [("实时概率（每月只用当时的数据）", list(q["增长加速"]), C.S[0]),
                                ("事后概率（全样本平滑）", list(post), C.MUTED)],
                        title="增长「加速」的实时概率和事后概率，2017-01 至 2026-09", ylo=0, yhi=1, yticks=[0, 0.5, 1], ref=0.5,
                        desc="实时概率频繁在 0 和 1 之间跳动；事后概率十年里只切换两次。", height=230)


def fig_pricing() -> str:
    p = read("attribution_16A_定价")
    ev = p[(p["口径"] == "实时确认的切换") & p["日期"].notna()]
    s = p[(p["口径"] == "实时确认的切换") & p["日期"].isna()].iloc[0]
    rows = [{"label": "确认之前 6 个月", "mean": float(s["股 之前 6 个月（均值）"]), "dots": list(ev["股 之前 6 个月"].astype(float)), "sub": f"t {s['股 之前 6 个月 t']:.1f}"},
            {"label": "确认之后 6 个月", "mean": float(s["股 之后 6 个月（均值）"]), "dots": list(ev["股 之后 6 个月"].astype(float)), "sub": f"t {s['股 之后 6 个月 t']:.1f}"}]
    return C.strip_plot(rows, title="实时确认增长切换前后，沪深300 按信号方向的累计超额收益", unit="%", axis_note="%，8 次切换；灰点是每一次，竖线是均值",
                        desc="确认之前股票已经按信号方向走了 3.7%，确认之后平均反向 5.5%；次数少，两个均值都不显著。")


def fig_mapping() -> str:
    m = read("attribution_16A_映射")
    m = m[m["分组"].str.startswith("参考周期")].set_index("象限")
    rows = [{"label": k, "value": float(m.loc[k, "债（月均超额 %）"]), "sub": f"{int(m.loc[k, '月数'])} 个月，t {m.loc[k, '债 t']:.1f}"} for k in ("衰退", "复苏", "过热", "滞胀")]
    return C.hbar(rows, title="按 PMI、PPI 参考周期分组，债券每月的超额收益", unit="%", nd=2, diverging=True, axis_note="%，中债 7–10 年国债财富指数减 3 个月 SHIBOR",
                  desc="衰退期 +0.24%（t 2.3），过热期 −0.29%（t −2.1），和美林时钟的排序一致。")


def fig_kuttner() -> str:
    e = read("policy_irs_events")
    k = e[e["分组"].isin(["首发", "跟随", "LPR 跟随", "LPR 单独", "降息（基准）", "加息（基准）"])].dropna(subset=["Δ互换1Y(bp)", "Δ10年(bp)"])
    x, y = k["Δ互换1Y(bp)"].astype(float), k["Δ10年(bp)"].astype(float)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    kt = read("policy_irs_kuttner").set_index(["反应", "部分"])
    b, t = kt.loc[("Δ10年(bp)", "意外(bp)"), "系数"], kt.loc[("Δ10年(bp)", "意外(bp)"), "t"]
    pts = [(float(a), float(c), f"{d} {g}：互换 {C.num(a, 1, True)}bp，10 年 {C.num(c, 1, True)}bp") for a, c, d, g in zip(x, y, k["date"], k["工具"])]
    return C.scatter_fit(pts, slope=float(fit.params.iloc[1]), intercept=float(fit.params.iloc[0]),
                         title="调息公告：利率互换度量的意外和 10 年国债的反应", xlabel="1 年期利率互换在公告窗口内的变化（bp，意外）",
                         ylabel="10 年国债收益率变化（bp）", fit_label=f"意外每 1bp → 10 年 {b:.2f}bp（t {t:.1f}）",
                         desc=f"{len(k)} 次调息公告。控制预期之内的部分后，意外每 1bp 带动 10 年国债 {b:.2f}bp。")


def fig_data_surprise() -> str:
    s = read("surprise_irs")
    s = s[s["样本"] == "去掉 2020–2021"].sort_values("每 1 个标准差 Δ互换1Y(bp)", ascending=False)
    rows, hi = [], set()
    for _, r in s.iterrows():
        rows.append({"label": r["指标"], "value": float(r["每 1 个标准差 Δ互换1Y(bp)"]), "sub": f"{int(r['次数'])} 次，打乱 p {r['打乱 p']:.2f}"})
        if r["打乱 p"] < 0.05:
            hi.add(r["指标"])
    return C.hbar(rows, title="数据每超预期 1 个标准差，1 年期利率互换当天动多少", unit="bp", nd=2, highlight=hi,
                  axis_note="bp，去掉 2020–2021 年；蓝色是打乱检验 p < 0.05", desc="增长类数据（社零、工业、固投）和 PPI 的意外显著推升短端利率。")


def fig_vol() -> str:
    v = read("volforecast_tests")
    v = v[v["组合"] == "80/20"].set_index("比较")
    names = [("市场基准 + 隐含波动", "隐含波动本身：市场基准 + 隐含波动"), ("+ 增长、通胀因子变化", "比较一：只加增长通胀因子变化"),
             ("+ 数据意外", "比较一：只加数据意外"), ("+ 政策公告", "比较一：只加政策公告"), ("+ 信贷脉冲", "比较一：只加信贷脉冲（2014 起）"),
             ("+ 宏观三项一起", "比较一：市场基准 + 宏观（三项一起）")]
    rows = [{"label": a, "value": float(v.loc[b, "样本外 R²"]) * 100, "sub": f"Clark-West t {v.loc[b, 'Clark-West t']:.2f}"} for a, b in names]
    return C.hbar(rows, title="预测固收+（债 80% + 股 20%）下月波动：相对市场基准的样本外 R²", unit="%", highlight={"市场基准 + 隐含波动"},
                  axis_note="%，基准是组合过去 1、3、12 个月的波动和股债各自的波动、相关", desc="只有期权隐含波动有显著增量（+5.5%）；宏观变量一项一项加、一起加，样本外都更差。")


def fig_attribution() -> str:
    a = read("attribution_16B_收益贡献")
    a = a[a["方案"] == "风险平价"].set_index("区间")
    names = ["黄金", "债券·利率变动", "债券·票息减现金", "股票"]
    cols = ["黄金", "债·利率变动等", "债·票息减现金", "股"]
    cats = [("全样本", "全样本 2017–2026"), ("前半段", "前半段 2017–2021"), ("后半段", "后半段 2021–2026")]
    matrix = [[float(a.loc[c, col]) for col in cols] for c, _ in cats]
    return C.stacked_hbar([n for _, n in cats], names, matrix, title="风险平价组合每年的超额收益从哪来", unit="pp",
                          axis_note="百分点/年，相对 3 个月 SHIBOR", desc="全样本每年 3.19 个百分点的超额收益里，黄金 1.60，债券 1.26，股票 0.33。")


def fig_gdp() -> str:
    f = read("nowcast_forecasts")
    row = f[(f["季度"] == "2026Q3") & (f["第几个月"] == 3)].iloc[0]
    rows = [("因子模型（我们的实时预测）", float(row["因子模型"]), True), ("上季度不变", float(row["上季度不变"]), False),
            ("市场一致预期（路透调查）", 4.4, False), ("AR(1)", float(row["AR(1)"]), False), ("工业增加值桥接", float(row["工业增加值桥接"]), False)]
    return C.dot_plot(rows, title="2026 年三季度 GDP：9 月末的预测和实际", ref=(4.5, "实际 4.5%"), unit="%", lo=3.8, hi=5.0,
                      desc="因子模型 4.0%，比实际低 0.5 个百分点；AR(1) 4.5%；一致预期 4.4%；工业增加值桥接 4.9%。")


def fig_lpr() -> str:
    g = read("policy_irs_groups").set_index("分组")
    order = [("首发降息", "首发·只看降息"), ("降准", "降准"), ("跟随降息（7 天逆回购 / MLF）", "跟随（7 天逆回购 / MLF）"), ("LPR 跟随下调", "LPR 跟随")]
    rows = []
    for lab, key in order:
        r = g.loc[key]
        t = "" if pd.isna(r["1Y t"]) else f"，t {r['1Y t']:.1f}"
        rows.append({"label": lab, "value": float(r["1Y 宽松方向平均(bp)"]), "sub": f"{int(r['1Y 次数'])} 次{t}"})
    return C.hbar(rows, title="不同政策公告前后，1 年期利率互换的平均变化", unit="bp", highlight={"首发降息", "LPR 跟随下调"},
                  axis_note="bp，公告前最后一个交易日到公告后第一个交易日", desc="首发降息平均 −7.5bp，跟在 MLF 后面的 LPR 下调平均 −0.3bp，不显著。")


def fig_curve_events() -> str:
    c = read("curve_q4_policy_events").set_index("事件")
    order = [("首发降息", "首发降息（7 天逆回购 / MLF）"), ("降息（2008–2015）", "降息（基准）"), ("降准", "降准"), ("LPR 跟随下调", "LPR 跟随")]
    labels = [a for a, _ in order]
    mark = lambda t: "*" if abs(t) >= 1.96 else ""
    level = [float(c.loc[k, "Δ水平(bp)"]) for _, k in order]
    slope = [float(c.loc[k, "Δ斜率(bp)"]) for _, k in order]
    return C.hbar_panels(labels, [("水平（整体高低）", level, [mark(c.loc[k, "Δ水平 t"]) for _, k in order]),
                                  ("斜率（长端减短端）", slope, [mark(c.loc[k, "Δ斜率 t"]) for _, k in order])],
                         title="政策公告前后，国债曲线三个因子中的水平和斜率怎么动", unit="bp", highlight_rows={0, 1, 2},
                         desc="降息时水平下移、斜率几乎不动；降准时斜率上升约 4.4bp，曲线变陡。星号表示 |t| ≥ 1.96。")


def fig_cgb_curve() -> str:
    cur = pd.concat([pd.read_csv(RAW / f"chinabond_curve_{y}.csv") for y in (2025, 2026)])
    cur = cur[cur["曲线名称"] == "中债国债收益率曲线"].set_index("日期")
    latest = {"日期": "2026-10-22", "3月": 1.1601, "6月": 1.1630, "1年": 1.1794, "3年": 1.2667, "5年": 1.3532, "7年": 1.4562, "10年": 1.6414, "30年": 2.064}
    cols = ["3月", "6月", "1年", "3年", "5年", "7年", "10年", "30年"]
    tenor = [0.25, 0.5, 1, 3, 5, 7, 10, 30]
    pick = lambda d: cur.loc[cur.index[cur.index <= d].max(), cols].astype(float).tolist()
    series = [("2026-10-22", [latest[c] for c in cols], C.S[0]), ("一个月前（" + cur.index[cur.index <= "2026-09-22"].max() + "）", pick("2026-09-22"), C.S[1]),
              ("一年前（" + cur.index[cur.index <= "2025-10-22"].max() + "）", pick("2025-10-22"), C.MUTED)]
    left, right, top, bottom = 34, C.W - 14, 44, 206
    X = lambda t: left + (np.sqrt(t) - np.sqrt(0.25)) / (np.sqrt(30) - np.sqrt(0.25)) * (right - left)
    allv = [v for _, vals, _ in series for v in vals]
    lo, hi = np.floor(min(allv) * 10) / 10 - 0.05, np.ceil(max(allv) * 10) / 10 + 0.05
    Y = lambda v: bottom - (v - lo) / (hi - lo) * (bottom - top)
    svg = C.Svg(232, "中债国债收益率曲线", "横轴按期限的平方根排布，方便看清短端。")
    C.legend(svg, [(n, c, "line") for n, _, c in series], y=10)
    for t in np.arange(np.ceil(lo * 5) / 5, hi, 0.2):
        svg.line(left, Y(t), right, Y(t), "grid")
        svg.text(left - 6, Y(t), f"{t:.1f}", size=C.TICK, cls="t3", anchor="end")
    svg.line(left, bottom, right, bottom, "base")
    for t, c in zip(tenor, ["3M", "", "1Y", "3Y", "5Y", "7Y", "10Y", "30Y"]):
        if c:
            svg.text(X(t), bottom + 13, c, size=C.TICK, cls="t3", anchor="middle")
    for name, vals, color in series:
        path = "M" + "L".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(tenor, vals))
        svg.add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for t, v, c in zip(tenor, vals, cols):
            svg.add(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3" fill="{color}" stroke="{C.SURFACE}" stroke-width="1.5"><title>{C.esc(name)} {c}：{v:.2f}%</title></circle>')
    svg.text(left - 6, top - 12, "%", size=C.TICK, cls="t3", anchor="end", baseline="auto")
    return svg.render()


if __name__ == "__main__":
    for name, fn in [("chain", fig_chain), ("pricing", fig_pricing), ("mapping", fig_mapping), ("kuttner", fig_kuttner),
                     ("data-surprise", fig_data_surprise), ("vol", fig_vol), ("attribution", fig_attribution), ("gdp-2026q3", fig_gdp),
                     ("lpr", fig_lpr), ("curve-events", fig_curve_events), ("cgb-curve", fig_cgb_curve), ("identify", fig_identify)]:
        save(name, fn())
