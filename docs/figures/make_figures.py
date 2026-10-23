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
        fill, stroke = ("#E3EAF2", C.INK) if strong else (C.SURFACE, C.BASE)
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
    return C.line_chart(dates, [("实时概率（每月只用当时的数据）", list(q["增长加速"]), C.INK),
                                ("事后概率（全样本平滑）", list(post), C.NEG)],
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
        color = {"首发降息": C.SURPRISE, "LPR 跟随下调": C.INK}.get(lab, C.DIM)
        rows.append({"label": lab, "value": float(r["1Y 宽松方向平均(bp)"]), "sub": f"{int(r['1Y 次数'])} 次{t}", "color": color})
    return C.hbar(rows, title="不同政策公告前后，1 年期利率互换的平均变化", unit="bp",
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
    series = [("2026-10-22", [latest[c] for c in cols], C.INK), ("一个月前（" + cur.index[cur.index <= "2026-09-22"].max() + "）", pick("2026-09-22"), C.STEEL),
              ("一年前（" + cur.index[cur.index <= "2025-10-22"].max() + "）", pick("2025-10-22"), "#A9BBCD")]
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




def _events_by_tenor():
    """降息、降准公告前后，国债曲线每个期限的平均变化（宽松方向，bp），窗口和第 11 步一样。"""
    sys.path.insert(0, str(PROJECT / "src"))
    from macrob import curve, policy, policy_rates
    c = curve.curve_daily()
    days = c.index
    old, rates = policy.events(), policy_rates.rate_events()
    groups = {"降息": pd.concat([old[old["type"] == "降息（基准）"][["date", "direction"]],
                               rates[(rates["分组"] == "首发") & (rates["direction"] < 0)][["date", "direction"]]]),
              "降准": old[old["type"] == "降准"][["date", "direction"]]}
    out = {}
    for name, e in groups.items():
        rows = []
        for d in e["date"]:
            a = days.searchsorted(d, side="left") - 1
            b = days.searchsorted(d, side="right")
            if a >= 0 and b < len(days):
                rows.append((c.iloc[b] - c.iloc[a]).to_numpy() * 100)
        out[name] = (np.mean(rows, axis=0), len(rows))
    return list(c.columns), out


def fig_cut_vs_rrr() -> str:
    cols, out = _events_by_tenor()
    tenors = [0.25, 0.5, 1, 3, 5, 7, 10, 30]
    (cut, n1), (rrr, n2) = out["降息"], out["降准"]
    return C.tenor_lines(tenors, [(f"降息（{n1} 次）", list(cut), C.INK, 2.4), (f"降准（{n2} 次）", list(rrr), C.S[0], 2.4)],
                         title="降息和降准公告前后，国债曲线各期限的平均变化", ylabel="bp", zero_label="不变",
                         desc="降息时各期限一起下行；降准时短端下行更多、长端几乎不动，曲线变陡。")


def fig_curve_pair() -> str:
    from hero import STATES, TENORS, NAMES
    W2, panel, gap, top, bottom = C.W, (C.W - 20) / 2, 20, 30, 196
    lo, hi = 1.35, 3.30
    svg = C.Svg(232, "同样是降息：首发那天和跟随那天的国债曲线", "左边首发降息，两条线之间是意外带来的下移；右边跟随的 LPR 下调，两条线几乎重合。")
    for p, (key, title) in enumerate((("first", "首发 2022-08-15"), ("follow", "跟随 LPR 2022-08-22"))):
        st = STATES[key]
        x0 = p * (panel + gap) + 30
        x1 = x0 + panel - 46
        X = lambda t: x0 + (np.sqrt(t) - np.sqrt(0.25)) / (np.sqrt(30) - np.sqrt(0.25)) * (x1 - x0)
        Y = lambda v: bottom - (v - lo) / (hi - lo) * (bottom - top)
        svg.text(x0, 12, title, size=C.LABEL, weight="600")
        for v in (1.5, 2.0, 2.5, 3.0):
            svg.line(x0, Y(v), x1, Y(v), "grid")
            if p == 0:
                svg.text(x0 - 6, Y(v), f"{v:.1f}", size=C.TICK, cls="t3", anchor="end")
        for t, n in zip(TENORS, NAMES):
            if n in ("3M", "1Y", "5Y", "10Y", "30Y"):
                svg.text(X(t), bottom + 14, n, size=C.TICK, cls="t3", anchor="middle")
        pre = "L".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(TENORS, st["pre"]))
        post = "L".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(TENORS, st["post"]))
        back = "L".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in reversed(list(zip(TENORS, st["post"]))))
        svg.add(f'<path d="M{pre}L{back}Z" fill="{C.SURPRISE if st["surprise"] else "#AFC0D2"}" fill-opacity="0.9"/>')
        svg.add(f'<path d="M{pre}" fill="none" stroke="{C.STEEL}" stroke-width="1.6"/><path d="M{post}" fill="none" stroke="{C.INK}" stroke-width="2.4"/>')
        d10 = (st["post"][6] - st["pre"][6]) * 100
        color = C.SURPRISE_TEXT if st["surprise"] else C.MUTED
        y_band = Y(min(st["pre"][6], st["post"][6])) + 3
        svg.add(f'<line x1="{X(10):.1f}" y1="{y_band:.1f}" x2="{X(10):.1f}" y2="{Y(1.95):.1f}" stroke="{color}" stroke-width="1"/>')
        svg.add(f'<text x="{X(10):.1f}" y="{Y(1.95) + 14:.1f}" font-size="{C.VALUE}" font-weight="700" fill="{color}" text-anchor="middle">'
                f'{"意外" if st["surprise"] else "预期之内"} {C.num(d10, 0, True, "bp")}</text>')
    svg.text(30, 224, "细线：公告前收盘　粗线：公告后收盘", size=C.TICK, cls="t3")
    return svg.render()


def fig_gdp_curve() -> str:
    tenors = [0.25, 0.5, 1, 3, 5, 7, 10, 30]
    before = [1.1568, 1.1587, 1.1703, 1.2729, 1.3815, 1.4700, 1.6480, 2.060]
    after = [1.1750, 1.1752, 1.1813, 1.2674, 1.3770, 1.4673, 1.6497, 2.061]
    chg = [(b - a) * 100 for a, b in zip(before, after)]
    return C.tenor_lines(tenors, [("10-16 → 10-19 收盘", chg, C.INK, 2.4)], title="GDP 公布当天，国债曲线各期限的变化", ylabel="bp", zero_label="不变",
                         height=220, desc="短端上行 1–2bp，中长端基本不动，都在平常一天的波动范围里。")


def fig_carry() -> str:
    c = pd.read_csv(pathlib.Path(__file__).parent / "data" / "carry_20261022.csv", encoding="utf-8-sig")
    labels = [f"{int(t)} 年" for t in c["期限（年）"]]
    return C.hbar_panels(labels, [("票息减资金成本", list(c["票息减资金成本（bp）"]), [""] * len(c)), ("骑乘", list(c["骑乘（bp）"]), [""] * len(c))],
                         title="国债各期限持有 3 个月：票息减资金成本和骑乘（2026-10-22）", unit="bp",
                         desc="1–3 年收益率低于资金成本，票息为负；7–10 年骑乘最厚。")


def mini_lpr() -> str:
    return C.mini_hbar([{"label": "首发降息", "value": -7.5, "color": C.SURPRISE}, {"label": "跟随的 LPR", "value": -0.3, "color": C.INK}],
                       title="首发降息和跟随的 LPR 下调，1 年期利率互换的平均变化", unit="bp")


def mini_brier() -> str:
    return C.mini_hbar([{"label": "增长概率", "value": 0.39}, {"label": "通胀概率", "value": 0.32}], title="实时状态概率的 Brier 分数（越低越好）",
                       nd=2, signed=False, ref=(0.25, "一直猜 50%"))


def mini_vol() -> str:
    return C.mini_hbar([{"label": "期权隐含波动", "value": 5.5, "color": C.INK}, {"label": "宏观变量", "value": -7.1, "color": C.DIM}],
                       title="预测固收+ 组合下月波动：相对市场基准的样本外 R²", unit="%")


def hero_fragment() -> str:
    from hero import fragment
    return fragment("me@example.com")


def fig_chain_narrow() -> str:
    """手机上用的竖排版本：四个方框从上到下，市场预期这一环强调。"""
    boxes = [("宏观数据", "按首次公布日对齐", False), ("宏观状态", "增长、通胀的实时概率", False),
             ("市场预期", "一致预期、利率互换、期权", True), ("资产配置", "风险预算、固收+", False)]
    w, h, gap, x = 300, 56, 18, 10
    svg = C.Svg(4 * h + 3 * gap + 40, "研究框架：宏观状态和资产配置之间，隔着市场预期这一环",
                "四个方框从上到下：宏观数据、宏观状态、市场预期、资产配置。市场预期这一环被强调。", width=320)
    for i, (head, sub, strong) in enumerate(boxes):
        y = 6 + i * (h + gap)
        fill, stroke = ("#E3EAF2", C.INK) if strong else (C.SURFACE, C.BASE)
        svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{1.5 if strong else 1}"/>')
        svg.text(x + 18, y + h / 2, head, size=16, weight="600")
        svg.text(x + w - 16, y + h / 2, sub, size=13, cls="t2", anchor="end")
        if i < 3:
            ay = y + h + 3
            svg.add(f'<path d="M{x + 40},{ay}v{gap - 6}" stroke="{C.MUTED}" stroke-width="1.4"/><path d="M{x + 36.5},{ay + gap - 9}l3.5,4l3.5,-4" fill="none" stroke="{C.MUTED}" stroke-width="1.4"/>')
    svg.text(160, 4 * h + 3 * gap + 28, "美林时钟从「宏观状态」直接跳到「资产配置」", size=12, cls="t3", anchor="middle")
    return svg.render()


if __name__ == "__main__":
    for name, fn in [("chain", fig_chain), ("chain-narrow", fig_chain_narrow), ("pricing", fig_pricing), ("mapping", fig_mapping), ("kuttner", fig_kuttner),
                     ("data-surprise", fig_data_surprise), ("vol", fig_vol), ("attribution", fig_attribution), ("gdp-2026q3", fig_gdp),
                     ("lpr", fig_lpr), ("curve-events", fig_curve_events), ("cgb-curve", fig_cgb_curve), ("identify", fig_identify),
                     ("cut-vs-rrr", fig_cut_vs_rrr), ("curve-pair", fig_curve_pair), ("gdp-curve", fig_gdp_curve), ("carry", fig_carry),
                     ("mini-lpr", mini_lpr), ("mini-brier", mini_brier), ("mini-vol", mini_vol)]:
        save(name, fn())
    (OUT / "hero.html").write_text(hero_fragment(), encoding="utf-8")
    print("写了 hero")
