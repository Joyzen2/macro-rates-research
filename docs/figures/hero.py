"""首页第一屏：同样是降息，首发那天整条曲线下移，跟随那天不动。生成 charts/hero.html（网页片段）。

数据：中债国债收益率曲线收盘，窗口和研究里一样，公告前最后一个交易日 → 公告后第一个交易日。
- 首发：2022-08-15 央行下调 7 天逆回购和 MLF 各 10bp，窗口 08-12 → 08-16
- 跟随：2022-08-22 LPR 下调，窗口 08-19 → 08-23
没有 JavaScript 时显示首发那一组；有的话，开关切换、鼠标或方向键逐个期限读数。
"""
from __future__ import annotations

import html
import json
import math

TENORS = [0.25, 0.5, 1, 3, 5, 7, 10, 30]
NAMES = ["3M", "6M", "1Y", "3Y", "5Y", "7Y", "10Y", "30Y"]
STATES = {
    "first": {
        "button": "首发 · 2022-08-15",
        "pre": [1.4508, 1.6614, 1.8115, 2.3358, 2.4892, 2.7072, 2.7347, 3.2422],
        "post": [1.4111, 1.5799, 1.7199, 2.2449, 2.3828, 2.6011, 2.6346, 3.1759],
        "preDate": "2022-08-12", "postDate": "2022-08-16", "surprise": True,
        "lead": "2022 年 8 月 15 日，央行把 7 天逆回购和 MLF 利率各下调 10bp。公告前后 1 年期利率互换下行 12bp，说明市场事先没有预期到；整条国债曲线跟着下移，10 年期下行 10bp。",
    },
    "follow": {
        "button": "跟随 · 2022-08-22",
        "pre": [1.4308, 1.5953, 1.7248, 2.2279, 2.3708, 2.5812, 2.5875, 3.1141],
        "post": [1.4604, 1.6315, 1.7571, 2.2719, 2.4091, 2.5971, 2.6075, 3.1105],
        "preDate": "2022-08-19", "postDate": "2022-08-23", "surprise": False,
        "lead": "一周后的 8 月 22 日，LPR 跟着下调。这一步早在预期之内，曲线没有再往下走，10 年期反而上行 2bp。",
    },
}
LO, HI = 1.35, 3.30
CHARTS = {"wide": {"w": 1200, "h": 470, "x0": 64, "x1": 1040, "y0": 410, "y1": 30, "font": 15},
          "narrow": {"w": 420, "h": 330, "x0": 44, "x1": 356, "y0": 270, "y1": 40, "font": 13}}
INK, STEEL, MUTED, GRID, BASE, SURFACE, SURPRISE, SURPRISE_TEXT, CALM = "#0F2340", "#5E7896", "#55697F", "#DDE3EA", "#A7B4C2", "#F4F6F8", "#D2462E", "#B5361F", "#AFC0D2"


def _xy(g):
    X = lambda t: g["x0"] + (math.sqrt(t) - math.sqrt(TENORS[0])) / (math.sqrt(TENORS[-1]) - math.sqrt(TENORS[0])) * (g["x1"] - g["x0"])
    Y = lambda v: g["y0"] - (v - LO) / (HI - LO) * (g["y0"] - g["y1"])
    return X, Y


def _path(g, vals):
    X, Y = _xy(g)
    return "M" + "L".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(TENORS, vals))


def _band(g, a, b):
    X, Y = _xy(g)
    back = "L".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in reversed(list(zip(TENORS, b))))
    return _path(g, a) + "L" + back + "Z"


def _chart(kind: str, st: dict) -> str:
    g = CHARTS[kind]
    X, Y = _xy(g)
    f = g["font"]
    parts = [f'<svg class="curve-chart curve-{kind}" viewBox="0 0 {g["w"]} {g["h"]}" data-kind="{kind}" tabindex="0" role="img" '
             f'aria-label="国债收益率曲线，公告前后两天的收盘。按左右方向键逐个期限读数。" xmlns="http://www.w3.org/2000/svg">']
    for v in (1.5, 2.0, 2.5, 3.0):
        parts.append(f'<line x1="{g["x0"]}" y1="{Y(v):.1f}" x2="{g["x1"]}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{g["x0"] - 10}" y="{Y(v):.1f}" font-size="{f - 2}" fill="{MUTED}" text-anchor="end" dominant-baseline="middle">{v:.1f}%</text>')
    ruler_y = g["y0"] + 18
    parts.append(f'<line x1="{g["x0"]}" y1="{ruler_y}" x2="{g["x1"]}" y2="{ruler_y}" stroke="{BASE}" stroke-width="1"/>')
    for t, n in zip(TENORS, NAMES):
        parts.append(f'<line x1="{X(t):.1f}" y1="{ruler_y}" x2="{X(t):.1f}" y2="{ruler_y + 7}" stroke="{BASE}" stroke-width="1"/>')
        if kind == "wide" or n not in ("6M",):
            parts.append(f'<text x="{X(t):.1f}" y="{ruler_y + 24}" font-size="{f - 1}" fill="{MUTED}" text-anchor="middle">{n}</text>')
    band_fill = SURPRISE if st["surprise"] else CALM
    parts.append(f'<path class="band" d="{_band(g, st["pre"], st["post"])}" fill="{band_fill}" fill-opacity="0.9"/>')
    parts.append(f'<path class="pre" d="{_path(g, st["pre"])}" fill="none" stroke="{STEEL}" stroke-width="{2 if kind == "wide" else 1.6}" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(f'<path class="post" d="{_path(g, st["post"])}" fill="none" stroke="{INK}" stroke-width="{3.2 if kind == "wide" else 2.4}" stroke-linejoin="round" stroke-linecap="round"/>')
    d10 = (st["post"][6] - st["pre"][6]) * 100
    lx, ly = X(10), Y(min(st["pre"][6], st["post"][6])) + (34 if kind == "wide" else 26)
    word = "意外" if st["surprise"] else "预期之内"
    color = SURPRISE_TEXT if st["surprise"] else MUTED
    parts.append(f'<line class="band-leader" x1="{lx:.1f}" y1="{Y(min(st["pre"][6], st["post"][6])) + 4:.1f}" x2="{lx:.1f}" y2="{ly - f:.1f}" stroke="{color}" stroke-width="1.2"/>')
    label = f"{word} {'+' if d10 >= 0 else '−'}{abs(d10):.0f}bp"
    parts.append(f'<text class="band-label" x="{lx:.1f}" y="{ly:.1f}" font-size="{f + 3}" font-weight="700" fill="{color}" text-anchor="middle">{label}</text>')
    if kind == "wide":
        ex = g["x1"] + 14
        parts.append(f'<text class="end-pre" x="{ex}" y="{Y(st["pre"][-1]):.1f}" font-size="{f - 1}" fill="{STEEL}" dominant-baseline="middle">公告前 {st["preDate"][5:]}</text>')
        parts.append(f'<text class="end-post" x="{ex}" y="{Y(st["post"][-1]) + 18:.1f}" font-size="{f - 1}" fill="{INK}" font-weight="600" dominant-baseline="middle">公告后 {st["postDate"][5:]}</text>')
    parts.append(f'<g class="tip" visibility="hidden"><line class="tip-line" x1="0" y1="{g["y1"] - 10}" x2="0" y2="{g["y0"] + 10}" stroke="{INK}" stroke-width="1"/>'
                 f'<circle class="tip-pre" r="4.5" fill="{SURFACE}" stroke="{STEEL}" stroke-width="2"/><circle class="tip-post" r="5" fill="{INK}" stroke="{SURFACE}" stroke-width="2"/>'
                 f'<rect class="tip-box" rx="2" fill="{INK}"/><text class="tip-t1" font-size="{f - 1}" fill="#FFFFFF" font-weight="600"></text>'
                 f'<text class="tip-t2" font-size="{f - 1}" fill="#DDE3EA"></text></g>')
    parts.append(f'<rect class="hit" x="{g["x0"] - 20}" y="{g["y1"] - 20}" width="{g["x1"] - g["x0"] + 40}" height="{g["y0"] - g["y1"] + 40}" fill="transparent"/>')
    parts.append("</svg>")
    return "".join(parts)


def _table() -> str:
    head = "".join(f"<th class=num>{n}</th>" for n in NAMES)
    rows = []
    for st in STATES.values():
        for key, date in (("pre", st["preDate"]), ("post", st["postDate"])):
            rows.append(f"<tr><th>{date} 收盘</th>" + "".join(f"<td class=num>{v:.2f}%</td>" for v in st[key]) + "</tr>")
        rows.append("<tr class=chg><th>变化</th>" + "".join(f"<td class=num>{(b - a) * 100:+.1f}bp</td>".replace("-", "−") for a, b in zip(st["pre"], st["post"])) + "</tr>")
    return f'<details class="data"><summary>看每个期限的数</summary><div class="table"><table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>'


SCRIPT = r"""
(function(){
  var data = JSON.parse(document.getElementById('hero-data').textContent);
  var T = data.tenors, N = data.names, charts = [].slice.call(document.querySelectorAll('.curve-chart'));
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var key = 'first', cur = { pre: data.states.first.pre.slice(), post: data.states.first.post.slice() }, idx = 6;
  function geo(svg) { return data.charts[svg.getAttribute('data-kind')]; }
  function X(g, t) { return g.x0 + (Math.sqrt(t) - Math.sqrt(T[0])) / (Math.sqrt(T[T.length - 1]) - Math.sqrt(T[0])) * (g.x1 - g.x0); }
  function Y(g, v) { return g.y0 - (v - data.lo) / (data.hi - data.lo) * (g.y0 - g.y1); }
  function path(g, vals) { return 'M' + vals.map(function(v, i){ return X(g, T[i]).toFixed(1) + ',' + Y(g, v).toFixed(1); }).join('L'); }
  function band(g, a, b) { var back = b.map(function(v, i){ return X(g, T[i]).toFixed(1) + ',' + Y(g, v).toFixed(1); }).reverse().join('L'); return path(g, a) + 'L' + back + 'Z'; }
  function fmt(x) { var s = (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(1); return s + 'bp'; }
  function draw() {
    var st = data.states[key];
    charts.forEach(function(svg){
      var g = geo(svg);
      svg.querySelector('.pre').setAttribute('d', path(g, cur.pre));
      svg.querySelector('.post').setAttribute('d', path(g, cur.post));
      svg.querySelector('.band').setAttribute('d', band(g, cur.pre, cur.post));
      var lo = Math.min(cur.pre[6], cur.post[6]), lx = X(g, 10), ly = Y(g, lo) + (g.w > 600 ? 34 : 26);
      var lab = svg.querySelector('.band-label'), lead = svg.querySelector('.band-leader');
      var d10 = (cur.post[6] - cur.pre[6]) * 100;
      lab.textContent = (st.surprise ? '意外 ' : '预期之内 ') + (d10 >= 0 ? '+' : '−') + Math.abs(d10).toFixed(0) + 'bp';
      lab.setAttribute('y', ly.toFixed(1)); lead.setAttribute('y1', (Y(g, lo) + 4).toFixed(1)); lead.setAttribute('y2', (ly - g.font).toFixed(1));
      lab.setAttribute('fill', st.surprise ? data.colors.surpriseText : data.colors.muted); lead.setAttribute('stroke', st.surprise ? data.colors.surpriseText : data.colors.muted);
      svg.querySelector('.band').setAttribute('fill', st.surprise ? data.colors.surprise : data.colors.calm);
      var ep = svg.querySelector('.end-pre'), eq = svg.querySelector('.end-post');
      if (ep) { ep.setAttribute('y', Y(g, cur.pre[7]).toFixed(1)); ep.textContent = '公告前 ' + st.preDate.slice(5); }
      if (eq) { eq.setAttribute('y', (Y(g, cur.post[7]) + 18).toFixed(1)); eq.textContent = '公告后 ' + st.postDate.slice(5); }
      if (svg.querySelector('.tip').getAttribute('visibility') === 'visible') tip(svg, idx);
    });
  }
  function go(next) {
    if (next === key) return;
    var from = { pre: cur.pre.slice(), post: cur.post.slice() }, to = data.states[next];
    key = next;
    document.getElementById('hero-lead').textContent = to.lead;
    [].forEach.call(document.querySelectorAll('.switch button'), function(b){ b.setAttribute('aria-pressed', b.getAttribute('data-state') === next ? 'true' : 'false'); });
    document.querySelector('.hero').setAttribute('data-state', next);
    var token = {}; go.token = token;
    function finish() { if (go.token !== token) return; cur = { pre: to.pre.slice(), post: to.post.slice() }; draw(); }
    draw();
    if (reduce) { finish(); return; }
    var t0 = performance.now(), dur = 650;
    function step(now) {
      if (go.token !== token) return;
      var p = Math.min(1, (now - t0) / dur), e = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);
      cur.pre = from.pre.map(function(v, i){ return v + (to.pre[i] - v) * e; });
      cur.post = from.post.map(function(v, i){ return v + (to.post[i] - v) * e; });
      draw();
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
    setTimeout(finish, dur + 80);
  }
  function tip(svg, i) {
    var g = geo(svg), st = data.states[key], t = svg.querySelector('.tip');
    var x = X(g, T[i]), yp = Y(g, st.pre[i]), yq = Y(g, st.post[i]);
    t.setAttribute('visibility', 'visible');
    t.querySelector('.tip-line').setAttribute('x1', x); t.querySelector('.tip-line').setAttribute('x2', x);
    t.querySelector('.tip-pre').setAttribute('cx', x); t.querySelector('.tip-pre').setAttribute('cy', yp);
    t.querySelector('.tip-post').setAttribute('cx', x); t.querySelector('.tip-post').setAttribute('cy', yq);
    var t1 = t.querySelector('.tip-t1'), t2 = t.querySelector('.tip-t2'), box = t.querySelector('.tip-box');
    t1.textContent = N[i] + '：' + st.pre[i].toFixed(2) + '% → ' + st.post[i].toFixed(2) + '%';
    t2.textContent = '变化 ' + fmt((st.post[i] - st.pre[i]) * 100);
    var w = g.font * 13, bx = x + 14 + w > g.w ? x - 14 - w : x + 14, by = Math.max(g.y1 - 6, Math.min(yp, yq) - 70);
    box.setAttribute('x', bx); box.setAttribute('y', by); box.setAttribute('width', w); box.setAttribute('height', g.font * 3.4);
    t1.setAttribute('x', bx + 12); t1.setAttribute('y', by + g.font * 1.35); t2.setAttribute('x', bx + 12); t2.setAttribute('y', by + g.font * 2.65);
  }
  function hide(svg) { svg.querySelector('.tip').setAttribute('visibility', 'hidden'); }
  function nearest(svg, ev) {
    var p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
    var q = p.matrixTransform(svg.getScreenCTM().inverse()), g = geo(svg), best = 0, d = 1e9;
    T.forEach(function(t, i){ var dd = Math.abs(X(g, t) - q.x); if (dd < d) { d = dd; best = i; } });
    return best;
  }
  charts.forEach(function(svg){
    var hit = svg.querySelector('.hit');
    hit.addEventListener('pointermove', function(ev){ idx = nearest(svg, ev); tip(svg, idx); });
    hit.addEventListener('pointerleave', function(){ hide(svg); });
    svg.addEventListener('focus', function(){ tip(svg, idx); });
    svg.addEventListener('blur', function(){ hide(svg); });
    svg.addEventListener('keydown', function(ev){
      if (ev.key === 'ArrowRight') { idx = Math.min(T.length - 1, idx + 1); tip(svg, idx); ev.preventDefault(); }
      if (ev.key === 'ArrowLeft') { idx = Math.max(0, idx - 1); tip(svg, idx); ev.preventDefault(); }
    });
  });
  [].forEach.call(document.querySelectorAll('.switch button'), function(b){ b.addEventListener('click', function(){ go(b.getAttribute('data-state')); }); });
  document.querySelector('.hero').classList.add('ready');
})();
"""


def fragment(email: str) -> str:
    st = STATES["first"]
    data = {"tenors": TENORS, "names": NAMES, "lo": LO, "hi": HI, "states": STATES,
            "charts": {k: {**v} for k, v in CHARTS.items()},
            "colors": {"surprise": SURPRISE, "surpriseText": SURPRISE_TEXT, "muted": MUTED, "calm": CALM}}
    buttons = "".join(f'<button type="button" data-state="{k}" aria-pressed="{"true" if k == "first" else "false"}">{html.escape(v["button"])}</button>' for k, v in STATES.items())
    return f"""<section class="hero" data-state="first" aria-labelledby="hero-title">
  <div class="hero-copy">
    <h1 id="hero-title" class="display">同样是降息，<br>市场只为<span class="surprise">意外</span>定价。</h1>
    <p class="lead" id="hero-lead">{html.escape(st["lead"])}</p>
  </div>
  <div class="hero-side">
    <div class="switch" role="group" aria-label="选一天看曲线">{buttons}</div>
    <div class="actions"><a class="btn primary" href="research/macro-vs-expectations/">读主线研究</a><a class="btn" href="mailto:{email}">发邮件</a></div>
  </div>
  <figure class="hero-figure">
    {_chart("wide", st)}{_chart("narrow", st)}
    <figcaption>中债国债收益率曲线收盘。窗口和研究里一样：公告前最后一个交易日到公告后第一个交易日。</figcaption>
    {_table()}
  </figure>
  <script type="application/json" id="hero-data">{json.dumps(data, ensure_ascii=False)}</script>
  <script>{SCRIPT}</script>
</section>"""
