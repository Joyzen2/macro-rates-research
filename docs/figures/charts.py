"""手写 SVG 的小图表库：横向条形、堆叠条形、散点加拟合线、折线、点图。

- 颜色按 dataviz 参考调色板（分类色的固定顺序、强调色 + 灰色、发散用蓝红）；文字只用墨色，不用系列色。
- 细标记：条形高 16、数据端 3 的圆角（基线端是直角）；线宽 2；点半径 4 带 2px 底色描边；网格是实线细线。
- 每个标记带 <title>，鼠标悬停能看到数值；嵌进网页时文字继承网页字体。
"""
from __future__ import annotations

import html
import math

W = 460
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEG, DIM = "#e34948", "#c3c2b7"
FONT = '-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif'
STYLE = (f"<style>.vz text{{font-family:{FONT};fill:{INK};}}.vz .t2{{fill:{INK2};}}.vz .t3{{fill:{MUTED};}}"
         f".vz .grid{{stroke:{GRID};stroke-width:1;}}.vz .base{{stroke:{BASE};stroke-width:1;}}</style>")
LABEL, TICK, VALUE = 13, 11.5, 12


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def num(v: float, nd: int = 1, signed: bool = False, unit: str = "") -> str:
    s = f"{abs(v):.{nd}f}"
    sign = "−" if v < 0 and float(s) != 0 else ("+" if signed and v > 0 and float(s) != 0 else "")
    return sign + s + unit


def text_width(s: str, size: float) -> float:
    return sum(size if ord(c) > 0x2E80 else size * 0.6 for c in str(s))


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = max(hi - lo, 1e-12)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def tick_label(t: float) -> str:
    return num(t, 0 if abs(t - round(t)) < 1e-9 else (1 if abs(t * 10 - round(t * 10)) < 1e-9 else 2))


class Svg:
    def __init__(self, height: float, title: str, desc: str = ""):
        self.h, self.title, self.desc, self.parts = height, title, desc, []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=LABEL, cls="", anchor="start", weight=None, baseline="middle") -> None:
        w = f' font-weight="{weight}"' if weight else ""
        c = f' class="{cls}"' if cls else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" dominant-baseline="{baseline}"{c}{w}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, cls="grid", extra="") -> None:
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"{extra}/>')

    def render(self) -> str:
        return (f'<svg class="vz" viewBox="0 0 {W} {self.h:.0f}" role="img" xmlns="http://www.w3.org/2000/svg">'
                f"<title>{esc(self.title)}</title><desc>{esc(self.desc)}</desc>{STYLE}{''.join(self.parts)}</svg>")


def hbar_path(x0: float, x1: float, y: float, h: float, r: float = 3.0) -> str:
    """横向条形：基线端直角，数据端圆角。"""
    r = min(r, abs(x1 - x0) / 2, h / 2)
    if x1 >= x0:
        return f"M{x0:.2f},{y:.2f}H{x1 - r:.2f}Q{x1:.2f},{y:.2f} {x1:.2f},{y + r:.2f}V{y + h - r:.2f}Q{x1:.2f},{y + h:.2f} {x1 - r:.2f},{y + h:.2f}H{x0:.2f}Z"
    return f"M{x0:.2f},{y:.2f}H{x1 + r:.2f}Q{x1:.2f},{y:.2f} {x1:.2f},{y + r:.2f}V{y + h - r:.2f}Q{x1:.2f},{y + h:.2f} {x1 + r:.2f},{y + h:.2f}H{x0:.2f}Z"


def _scale(lo: float, hi: float, x0: float, x1: float):
    return lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)


def legend(svg: Svg, items: list[tuple[str, str, str]], y: float, x: float = 0.0) -> None:
    """items：（名字，颜色，'box' 或 'line'）。"""
    for name, color, kind in items:
        if kind == "line":
            svg.add(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 16:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        else:
            svg.add(f'<rect x="{x:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
        svg.text(x + (21 if kind == "line" else 15), y, name, size=VALUE, cls="t2")
        x += (21 if kind == "line" else 15) + text_width(name, VALUE) + 16


def hbar(rows: list[dict], *, title: str, unit: str = "", nd: int = 1, signed: bool = True, diverging: bool = False,
         highlight: set[str] | None = None, desc: str = "", axis_note: str = "") -> str:
    """rows：label、value，可选 sub（标签下方的小字）、dots（同一行上要画的单个事件）、color。"""
    label_w = max(max(text_width(r["label"], LABEL), text_width(r.get("sub", ""), TICK)) for r in rows) + 12
    values = [r["value"] for r in rows] + [d for r in rows for d in r.get("dots", [])]
    vmin, vmax = min(0.0, *values), max(0.0, *values)
    lab = lambda r: num(r["value"], nd, signed, unit) + (r.get("mark") or "")
    left_m = max([text_width(lab(r), VALUE) + 10 for r in rows if r["value"] < 0] or [0])
    right_m = max([text_width(lab(r), VALUE) + 10 for r in rows if r["value"] >= 0] or [0])
    row_h, top = 30, 8
    plot_x0, plot_x1 = label_w, W - 4
    k = (plot_x1 - plot_x0 - left_m - right_m) / ((vmax - vmin) or 1.0)
    lo, hi = vmin - left_m / k, vmax + right_m / k
    ticks = [t for t in nice_ticks(vmin, vmax) if lo - 1e-9 <= t <= hi + 1e-9]
    x = _scale(lo, hi, plot_x0, plot_x1)
    height = top + row_h * len(rows) + 30 + (14 if axis_note else 0)
    svg = Svg(height, title, desc)
    bottom = top + row_h * len(rows)
    for t in ticks:
        svg.line(x(t), top, x(t), bottom, "base" if abs(t) < 1e-12 else "grid")
        svg.text(x(t), bottom + 12, tick_label(t), size=TICK, cls="t3", anchor="middle")
    if axis_note:
        svg.text(plot_x1, bottom + 28, axis_note, size=TICK, cls="t3", anchor="end")
    for i, r in enumerate(rows):
        yc = top + row_h * i + row_h / 2
        v = r["value"]
        if r.get("color"):
            color = r["color"]
        elif highlight is not None:
            color = S[0] if r["label"] in highlight else DIM
        elif diverging:
            color = S[0] if v >= 0 else NEG
        else:
            color = S[0]
        if r.get("sub"):
            svg.text(label_w - 12, yc - 6, r["label"], anchor="end")
            svg.text(label_w - 12, yc + 8, r["sub"], size=TICK, cls="t3", anchor="end")
        else:
            svg.text(label_w - 12, yc, r["label"], anchor="end")
        tip = f'{r["label"]}：{num(v, nd, signed, unit)}'
        svg.add(f'<path d="{hbar_path(x(0), x(v), yc - 8, 16)}" fill="{color}"><title>{esc(tip)}</title></path>')
        for d in r.get("dots", []):
            svg.add(f'<circle cx="{x(d):.1f}" cy="{yc:.1f}" r="3" fill="{SURFACE}" stroke="{INK2}" stroke-width="1.2"><title>{esc(num(d, nd, signed, unit))}</title></circle>')
        label = num(v, nd, signed, unit) + (r.get("mark") or "")
        if v >= 0:
            svg.text(x(v) + 6, yc, label, size=VALUE, cls="t2")
        else:
            svg.text(x(v) - 6, yc, label, size=VALUE, cls="t2", anchor="end")
    return svg.render()


def stacked_hbar(cats: list[str], names: list[str], matrix: list[list[float]], *, title: str, unit: str = "", nd: int = 2,
                 desc: str = "", axis_note: str = "") -> str:
    """每一行是一个类别，各段按 names 的顺序从 0 往外堆；负数往左堆。"""
    label_w = max(text_width(c, LABEL) for c in cats) + 12
    pos = [sum(v for v in row if v > 0) for row in matrix]
    neg = [sum(v for v in row if v < 0) for row in matrix]
    vmin, vmax = min(0.0, *neg), max(pos)
    right_m = max(text_width("合计 " + num(sum(r), nd, True, unit), VALUE) for r in matrix) + 10
    k = (W - 4 - label_w - right_m) / (vmax - vmin)
    lo, hi = vmin, vmax + right_m / k
    ticks = [t for t in nice_ticks(vmin, vmax) if lo - 1e-9 <= t <= hi + 1e-9]
    top, row_h = 34, 34
    x = _scale(lo, hi, label_w, W - 4)
    bottom = top + row_h * len(cats)
    svg = Svg(bottom + 30 + (14 if axis_note else 0), title, desc)
    legend(svg, [(n, S[i], "box") for i, n in enumerate(names)], y=10)
    for t in ticks:
        svg.line(x(t), top, x(t), bottom, "base" if abs(t) < 1e-12 else "grid")
        svg.text(x(t), bottom + 12, tick_label(t), size=TICK, cls="t3", anchor="middle")
    if axis_note:
        svg.text(W - 4, bottom + 28, axis_note, size=TICK, cls="t3", anchor="end")
    for i, (c, row) in enumerate(zip(cats, matrix)):
        yc = top + row_h * i + row_h / 2
        svg.text(label_w - 12, yc, c, anchor="end")
        right, left = 0.0, 0.0
        segs = []
        for j, v in enumerate(row):
            if v >= 0:
                segs.append((j, right, right + v)); right += v
            else:
                segs.append((j, left, left + v)); left += v
        last_pos = max((k for k, (j, a, b) in enumerate(segs) if b > a), key=lambda k: segs[k][2], default=None)
        for k, (j, a, b) in enumerate(segs):
            if abs(b - a) < 1e-12:
                continue
            xa, xb = x(a), x(b)
            gap = 1.0
            if b > a:
                xa = xa + (gap if a > 0 else 0)
                path = hbar_path(xa, xb, yc - 8, 16, 3 if k == last_pos else 0)
            else:
                xa = xa - (gap if a < 0 else 0)
                path = hbar_path(xa, xb, yc - 8, 16, 0)
            tip = f"{c} · {names[j]}：{num(row[j], nd, True, unit)}"
            svg.add(f'<path d="{path}" fill="{S[j]}"><title>{esc(tip)}</title></path>')
        svg.text(x(right) + 6, yc, "合计 " + num(sum(row), nd, True, unit), size=VALUE, cls="t2")
    return svg.render()


def scatter_fit(points: list[tuple[float, float, str]], *, slope: float, intercept: float, title: str, xlabel: str, ylabel: str,
                fit_label: str, desc: str = "", height: float = 300) -> str:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xlo, xhi = min(0.0, *xs), max(0.0, *xs)
    ylo, yhi = min(0.0, *ys), max(0.0, *ys)
    xp, yp = (xhi - xlo) * 0.06, (yhi - ylo) * 0.08
    xlo, xhi, ylo, yhi = xlo - xp, xhi + xp, ylo - yp, yhi + yp
    left, right, top, bottom = 40, W - 10, 14, height - 40
    X, Y = _scale(xlo, xhi, left, right), _scale(ylo, yhi, bottom, top)
    svg = Svg(height, title, desc)
    for t in nice_ticks(xlo, xhi, 6):
        if xlo <= t <= xhi:
            svg.line(X(t), top, X(t), bottom, "base" if abs(t) < 1e-12 else "grid")
            svg.text(X(t), bottom + 12, tick_label(t), size=TICK, cls="t3", anchor="middle")
    for t in nice_ticks(ylo, yhi, 5):
        if ylo <= t <= yhi:
            svg.line(left, Y(t), right, Y(t), "base" if abs(t) < 1e-12 else "grid")
            svg.text(left - 6, Y(t), tick_label(t), size=TICK, cls="t3", anchor="end")
    svg.text(right, bottom + 28, xlabel, size=TICK, cls="t3", anchor="end")
    svg.text(left, top - 4, ylabel, size=TICK, cls="t3", baseline="auto")
    xa, xb = min(xs), max(xs)
    svg.add(f'<line x1="{X(xa):.1f}" y1="{Y(intercept + slope * xa):.1f}" x2="{X(xb):.1f}" y2="{Y(intercept + slope * xb):.1f}" stroke="{S[0]}" stroke-width="2" stroke-linecap="round"/>')
    for x_, y_, tip in points:
        svg.add(f'<circle cx="{X(x_):.1f}" cy="{Y(y_):.1f}" r="4" fill="{S[0]}" fill-opacity="0.85" stroke="{SURFACE}" stroke-width="2"><title>{esc(tip)}</title></circle>')
    svg.add(f'<line x1="{left + 8:.1f}" y1="{top + 12:.1f}" x2="{left + 24:.1f}" y2="{top + 12:.1f}" stroke="{S[0]}" stroke-width="2" stroke-linecap="round"/>')
    svg.text(left + 30, top + 12, fit_label, size=VALUE, cls="t2")
    return svg.render()


def strip_plot(rows: list[dict], *, title: str, unit: str = "", nd: int = 1, desc: str = "", axis_note: str = "") -> str:
    """次数少、噪音大的事件研究：每行画出每一次事件的点，再用竖线标出均值。rows：label、sub、mean、dots。"""
    label_w = max(max(text_width(r["label"], LABEL), text_width(r.get("sub", ""), TICK)) for r in rows) + 12
    vals = [d for r in rows for d in r["dots"]] + [r["mean"] for r in rows] + [0.0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    top, row_h = 8, 44
    x = _scale(lo, hi, label_w, W - 8)
    bottom = top + row_h * len(rows)
    svg = Svg(bottom + 30 + (14 if axis_note else 0), title, desc)
    for t in nice_ticks(lo, hi):
        if lo <= t <= hi:
            svg.line(x(t), top, x(t), bottom, "base" if abs(t) < 1e-12 else "grid")
            svg.text(x(t), bottom + 12, tick_label(t), size=TICK, cls="t3", anchor="middle")
    if axis_note:
        svg.text(W - 8, bottom + 28, axis_note, size=TICK, cls="t3", anchor="end")
    for i, r in enumerate(rows):
        yc = top + row_h * i + row_h / 2 + 4
        svg.text(label_w - 12, yc - 7, r["label"], anchor="end")
        if r.get("sub"):
            svg.text(label_w - 12, yc + 8, r["sub"], size=TICK, cls="t3", anchor="end")
        for d in r["dots"]:
            svg.add(f'<circle cx="{x(d):.1f}" cy="{yc:.1f}" r="4" fill="{MUTED}" stroke="{SURFACE}" stroke-width="2"><title>{esc(num(d, nd, True, unit))}</title></circle>')
        m = r["mean"]
        color = S[0] if m >= 0 else NEG
        svg.add(f'<rect x="{x(m) - 1.5:.1f}" y="{yc - 11:.1f}" width="3" height="22" rx="1.5" fill="{color}"><title>均值 {esc(num(m, nd, True, unit))}</title></rect>')
        svg.text(x(m), yc - 18, "均值 " + num(m, nd, True, unit), size=TICK, cls="t2", anchor="middle")
    return svg.render()


def line_chart(dates: list, series: list[tuple[str, list[float], str]], *, title: str, ylo: float, yhi: float, yticks: list[float],
               ylabel: str = "", desc: str = "", height: float = 240, ref: float | None = None, year_step: int = 2) -> str:
    """dates 是 pandas 时间戳列表；series：（名字，数值，颜色）。"""
    left, right, top, bottom = 34, W - 8, 30, height - 26
    t0, t1 = dates[0].toordinal(), dates[-1].toordinal()
    X, Y = _scale(t0, t1, left, right), _scale(ylo, yhi, bottom, top)
    svg = Svg(height, title, desc)
    legend(svg, [(n, c, "line") for n, _, c in series], y=10)
    for t in yticks:
        svg.line(left, Y(t), right, Y(t), "grid")
        svg.text(left - 6, Y(t), tick_label(t), size=TICK, cls="t3", anchor="end")
    if ref is not None:
        svg.line(left, Y(ref), right, Y(ref), "base")
    y0, y1 = dates[0].year, dates[-1].year
    for yr in range(y0 + (1 if dates[0].month > 1 else 0), y1 + 1):
        if (yr - y0) % year_step:
            continue
        import datetime as _dt
        xt = X(_dt.date(yr, 1, 1).toordinal())
        svg.line(xt, bottom, xt, bottom + 4, "base")
        svg.text(xt, bottom + 14, str(yr), size=TICK, cls="t3", anchor="middle")
    svg.line(left, bottom, right, bottom, "base")
    for name, vals, color in series:
        pts = [(X(d.toordinal()), Y(v)) for d, v in zip(dates, vals) if v == v]
        path = "M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        svg.add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"><title>{esc(name)}</title></path>')
    if ylabel:
        svg.text(left, top - 6, ylabel, size=TICK, cls="t3", baseline="auto")
    return svg.render()


def dot_plot(rows: list[tuple[str, float, bool]], *, title: str, ref: tuple[float, str], unit: str = "", nd: int = 1,
             desc: str = "", lo: float | None = None, hi: float | None = None) -> str:
    """rows：（标签，数值，是否强调）；ref：（参考值，参考线的说明），比如实际公布值。"""
    label_w = max(text_width(r[0], LABEL) for r in rows) + 12
    vals = [r[1] for r in rows] + [ref[0]]
    lo = min(vals) - 0.25 if lo is None else lo
    hi = max(vals) + 0.25 if hi is None else hi
    top, row_h = 24, 28
    x = _scale(lo, hi, label_w, W - 30)
    bottom = top + row_h * len(rows)
    svg = Svg(bottom + 26, title, desc)
    for t in nice_ticks(lo, hi, 5):
        if lo <= t <= hi:
            svg.line(x(t), top, x(t), bottom, "grid")
            svg.text(x(t), bottom + 12, tick_label(t) + unit, size=TICK, cls="t3", anchor="middle")
    svg.add(f'<line x1="{x(ref[0]):.1f}" y1="{top - 6:.1f}" x2="{x(ref[0]):.1f}" y2="{bottom:.1f}" stroke="{INK}" stroke-width="1.2"/>')
    svg.text(x(ref[0]), top - 12, ref[1], size=VALUE, anchor="middle")
    for i, (label, v, strong) in enumerate(rows):
        yc = top + row_h * i + row_h / 2
        svg.text(label_w - 12, yc, label, anchor="end")
        svg.line(label_w, yc, W - 30, yc, "grid")
        color = S[0] if strong else INK2
        svg.add(f'<circle cx="{x(v):.1f}" cy="{yc:.1f}" r="5" fill="{color}" stroke="{SURFACE}" stroke-width="2"><title>{esc(label)}：{num(v, nd, False, unit)}</title></circle>')
        svg.text(x(v) + (10 if v >= ref[0] else -10), yc, num(v, nd, False, unit), size=VALUE, cls="t2", anchor="start" if v >= ref[0] else "end")
    return svg.render()


def hbar_panels(labels: list[str], panels: list[tuple[str, list[float], list[str]]], *, title: str, unit: str = "", nd: int = 1,
                desc: str = "", highlight_rows: set[int] | None = None) -> str:
    """几个横向条形图并排，共用左边的标签；panels：（小标题，数值，每个数值后面的标注）。"""
    label_w = max(text_width(l, LABEL) for l in labels) + 14
    gap, top, row_h = 14, 30, 28
    pw = (W - label_w - gap * (len(panels) - 1)) / len(panels)
    allv = [v for _, vals, _ in panels for v in vals]
    vmin, vmax = min(0.0, *allv), max(0.0, *allv)
    labs = [(v, num(v, nd, True, "") + m) for _, vals, marks in panels for v, m in zip(vals, marks)]
    left_m = max([text_width(t, TICK) + 6 for v, t in labs if v < 0] or [0])
    right_m = max([text_width(t, TICK) + 6 for v, t in labs if v >= 0] or [0])
    kk = (pw - left_m - right_m) / ((vmax - vmin) or 1.0)
    lo, hi = vmin - left_m / kk, vmax + right_m / kk
    bottom = top + row_h * len(labels)
    svg = Svg(bottom + 24, title, desc)
    for i, l in enumerate(labels):
        svg.text(label_w - 12, top + row_h * i + row_h / 2, l, anchor="end")
    for p, (ptitle, vals, marks) in enumerate(panels):
        x0 = label_w + p * (pw + gap)
        x = _scale(lo, hi, x0, x0 + pw)
        svg.text(x0 + pw / 2, 12, ptitle, size=LABEL, anchor="middle", weight="600")
        for t in nice_ticks(vmin, vmax, 4):
            if lo <= t <= hi:
                svg.line(x(t), top, x(t), bottom, "base" if abs(t) < 1e-12 else "grid")
                svg.text(x(t), bottom + 12, tick_label(t), size=TICK, cls="t3", anchor="middle")
        for i, (v, m) in enumerate(zip(vals, marks)):
            yc = top + row_h * i + row_h / 2
            color = S[0] if (highlight_rows is None or i in highlight_rows) else DIM
            svg.add(f'<path d="{hbar_path(x(0), x(v), yc - 7, 14)}" fill="{color}"><title>{esc(labels[i])} · {esc(ptitle)}：{num(v, nd, True, unit)}</title></path>')
            lab = num(v, nd, True, "") + m
            if v >= 0:
                svg.text(x(v) + 4, yc, lab, size=TICK, cls="t2")
            else:
                svg.text(x(v) - 4, yc, lab, size=TICK, cls="t2", anchor="end")
    return svg.render()
