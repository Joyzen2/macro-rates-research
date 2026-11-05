"""第二篇：国债期货四个期限对央行政策事件的反应。

uv run python -m macrob.rpt_bond_futures
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rpt_common import BOND, dominant, load, load_futures, mean_t, price_panel, random_window_placebo, save_csv, save_svg

REPORT = "bond-futures-policy"
START = pd.Timestamp("2019-01-01")
TENOR = {"TS": "中国国债收益率2年", "TF": "中国国债收益率5年", "T": "中国国债收益率10年", "TL": "中国国债收益率30年"}
GROUPS = ["首发", "跟随", "LPR 跟随", "发布会", "降准"]
DUR_WIN, DUR_MIN, AVOID = 250, 60, 10
DRIFT_H = [5, 10, 20]


class Product:
    """一个品种的交易日历、主力合约、价格面板、连续日收益和对应期限的收益率日变化。"""

    def __init__(self, fut, yields, prod):
        dom = dominant(fut, prod)
        dom = dom[dom.index >= START]
        self.name, self.cal = prod, dom.index
        self.dom = dom["contract"]
        self.px = price_panel(fut, prod).reindex(self.cal)
        prev = self.dom.shift(1)
        r = [np.nan] + [self.px.at[t, c] / self.px.at[tp, c] * 100 - 100 if isinstance(c, str) else np.nan
                        for t, tp, c in zip(self.cal[1:], self.cal[:-1], prev.iloc[1:])]
        self.ret = pd.Series(r, index=self.cal)                              # %，前一天的主力
        y = yields[TENOR[prod]].reindex(self.cal)
        self.dy = y.diff() * 100                                             # bp
        self.y = y

    def pos(self, d):
        return int(self.cal.get_loc(d))

    def window(self, d):
        """公告日前最后一个交易日 → 公告日后第一个交易日。"""
        before, after = self.cal[self.cal < d], self.cal[self.cal > d]
        if not len(before) or not len(after):
            return None
        return before[-1], after[0]

    def change(self, s, e, contract=None):
        c = contract or self.dom[s]
        return (self.px.at[e, c] / self.px.at[s, c] - 1) * 100

    def duration(self, s):
        """经验久期：截至 s 的过去 250 个交易日（至少 60 天），日收益（%）对收益率变化（bp）回归，斜率取相反数。单位 %/bp。"""
        i = self.pos(s)
        r, dy = self.ret.iloc[max(0, i - DUR_WIN + 1): i + 1], self.dy.iloc[max(0, i - DUR_WIN + 1): i + 1]
        ok = r.notna() & dy.notna()
        if ok.sum() < DUR_MIN:
            return np.nan
        b = np.polyfit(dy[ok], r[ok], 1)[0]
        return -b


def event_positions(cal, dates):
    """事件在交易日历上的位置：不是交易日的取下一个交易日。"""
    return np.unique(np.searchsorted(cal.values, pd.DatetimeIndex(dates).values))


def placebo_pool(p: Product, width: int, ev_pos, start_offset: int = 0):
    """所有可用的 width 期窗口的价格变化（%），用窗口起点的主力；窗口前后 10 个交易日内不能有任何政策事件。
    start_offset>0 用于事件之后的漂移：窗口从 i 开始，但避让的范围按 [i − start_offset − 10, i + width + 10] 算。"""
    out = []
    n = len(p.cal)
    for i in range(n - width):
        lo, hi = i - start_offset - AVOID, i + width + AVOID
        if lo < 0:
            continue
        if np.any((ev_pos >= lo) & (ev_pos <= hi)):
            continue
        v = p.change(p.cal[i], p.cal[i + width])
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def build_events(products, events):
    rows = []
    ev = events[(events["分组"].isin(GROUPS)) & (events.index >= START)]
    for d, e in ev.iterrows():
        sign = -e["direction"]                                   # 折成宽松方向：宽松 +1，收紧 −1
        for prod, p in products.items():
            w = p.window(d)
            if w is None or d < p.cal[0]:
                continue
            s, en = w
            dur = p.duration(s)
            chg = p.change(s, en)
            spot = (p.y[en] - p.y[s]) * 100
            rows.append({"date": d.date(), "group": e["分组"], "tool": e["工具"], "product": prod, "start": s.date(),
                         "end": en.date(), "sign": sign, "width": p.pos(en) - p.pos(s), "contract": p.dom[s], "duration": dur,
                         "duration_years": dur * 100, "price_chg": chg * sign,
                         "conv_bp": -chg * sign / dur if np.isfinite(dur) else np.nan,
                         "spot_bp": spot * sign, "usable": bool(np.isfinite(dur) and np.isfinite(chg))})
    return pd.DataFrame(rows)


def group_table(ev, products, ev_pos):
    rows = []
    pools = {}
    for (prod, g), h in ev[ev.usable].groupby(["product", "group"]):
        p = products[prod]
        widths = h["width"].tolist()
        for w in set(widths):
            if (prod, w) not in pools:
                pools[(prod, w)] = placebo_pool(p, w, ev_pos[prod])
        pp, center = random_window_placebo({w: pools[(prod, w)] for w in set(widths)}, widths, h.price_chg.mean())
        rows.append({"product": prod, "group": g, "n": len(h), "price_chg": h.price_chg.mean(), "t_price": mean_t(h.price_chg),
                     "conv_bp": h.conv_bp.mean(), "t_bp": mean_t(h.conv_bp), "placebo_p": pp, "placebo_center": center,
                     "spot_bp": h.spot_bp.mean(), "t_spot": mean_t(h.spot_bp)})
    t = pd.DataFrame(rows)
    t["g_order"] = t.group.map({g: i for i, g in enumerate(GROUPS)})
    t["p_order"] = t["product"].map({g: i for i, g in enumerate(BOND)})
    return t.sort_values(["p_order", "g_order"]).drop(columns=["g_order", "p_order"])


def vs_spot(ev):
    rows = []
    for prod, h in ev[ev.usable & ev.spot_bp.notna()].groupby("product"):
        rows.append({"product": prod, "n": len(h), "corr": np.corrcoef(h.conv_bp, h.spot_bp)[0, 1],
                     "mean_abs_diff": (h.conv_bp - h.spot_bp).abs().mean(), "mean_diff": (h.conv_bp - h.spot_bp).mean(),
                     "conv_mean": h.conv_bp.mean(), "spot_mean": h.spot_bp.mean(),
                     "same_sign": float(np.mean(np.sign(h.conv_bp) == np.sign(h.spot_bp)))})
    t = pd.DataFrame(rows)
    return t.set_index("product").loc[[p for p in BOND if p in set(t["product"])]].reset_index()


def curve(ev):
    """10 年减 2 年（T 对 TS）、30 年减 10 年（TL 对 T）的变化，和久期中性做陡组合（多短端、空长端）在窗口里的收益。"""
    piv = ev[ev.usable].pivot_table(index=["date", "group"], columns="product",
                                    values=["conv_bp", "spot_bp", "price_chg", "duration"])
    rows = []
    for (d, g), r in piv.iterrows():
        for name, short, long_ in [("10年−2年", "TS", "T"), ("30年−10年", "T", "TL")]:
            if not (np.isfinite(r.get(("conv_bp", short), np.nan)) and np.isfinite(r.get(("conv_bp", long_), np.nan))):
                continue
            ds, dl = r[("duration", short)], r[("duration", long_)]
            rows.append({"date": d, "group": g, "spread": name,
                         "conv_bp": r[("conv_bp", long_)] - r[("conv_bp", short)],
                         "spot_bp": r[("spot_bp", long_)] - r[("spot_bp", short)],
                         "hedge_ratio": ds / dl,
                         "steepener_ret": r[("price_chg", short)] - ds / dl * r[("price_chg", long_)]})
    per = pd.DataFrame(rows)
    out = []
    for (name, g), h in per.groupby(["spread", "group"]):
        out.append({"spread": name, "group": g, "n": len(h), "conv_bp": h.conv_bp.mean(), "t_conv": mean_t(h.conv_bp),
                    "spot_bp": h.spot_bp.mean(), "t_spot": mean_t(h.spot_bp),
                    "steepener_ret": h.steepener_ret.mean(), "t_steep": mean_t(h.steepener_ret)})
    t = pd.DataFrame(out)
    t["o"] = t.group.map({g: i for i, g in enumerate(GROUPS)})
    return per, t.sort_values(["spread", "o"]).drop(columns="o")


def drift(ev, p: Product, ev_pos):
    """首发事件后第 5、10、20 个交易日，T 相对窗口终点的累计价格变化（同一合约，折成宽松方向）。"""
    first = ev[(ev["product"] == p.name) & (ev.group == "首发")]
    rows, summ = [], []
    for r in first.itertuples():
        e = pd.Timestamp(r.end)
        i = p.pos(e)
        rec = {"date": r.date, "end": r.end, "contract": r.contract}
        for h in DRIFT_H:
            if i + h < len(p.cal):
                rec[f"h{h}"] = p.change(e, p.cal[i + h], r.contract) * r.sign   # 折成宽松方向
            else:
                rec[f"h{h}"] = np.nan
        rows.append(rec)
    per = pd.DataFrame(rows)
    for h in DRIFT_H:
        v = per[f"h{h}"].dropna()
        pool = placebo_pool(p, h, ev_pos, start_offset=0)
        pp, center = random_window_placebo({h: pool}, [h] * len(v), v.mean())
        summ.append({"product": p.name, "horizon": h, "n": len(v), "mean": v.mean(), "t": mean_t(v), "placebo_p": pp,
                     "placebo_center": center, "share_positive": float((v > 0).mean())})
    return per, pd.DataFrame(summ)


def case_1030(products):
    rows = []
    s, e = pd.Timestamp("2026-10-30"), pd.Timestamp("2026-11-02")
    for prod, p in products.items():
        c = p.dom[s]
        dur = p.duration(s)
        chg = p.change(s, e, c)
        rows.append({"product": prod, "contract": c, "close_1030": p.px.at[s, c], "close_1102": p.px.at[e, c],
                     "price_diff": p.px.at[e, c] - p.px.at[s, c], "price_chg": chg, "duration_years": dur * 100,
                     "conv_bp": -chg / dur})
    return pd.DataFrame(rows)


def charts(gt, cv, ev, spot_t):
    from charts import hbar_panels, strip_plot
    show = ["首发", "LPR 跟随", "降准"]
    panels = []
    for g in show:
        h = gt[gt.group == g].set_index("product").reindex(BOND)
        marks = ["*" if (pd.notna(p) and p < 0.05) else "" for p in h.placebo_p]
        panels.append((g, h.conv_bp.round(1).tolist(), marks))
    save_svg(hbar_panels(BOND, panels, title="事件窗口里，期货价格变化折算的收益率变化", unit="bp", nd=1, dim_unmarked=True,
                         tick_labels=False,
                         desc="宽松方向，bp；负数是收益率下行。星号是随机窗口安慰剂 p<0.05，淡色是不显著。"), "rpt3-groups")

    rows = []
    for prod in BOND:
        h = ev[(ev["product"] == prod) & (ev.group == "首发") & ev.usable]
        if len(h):
            rows.append({"label": prod, "sub": f"{len(h)} 次", "mean": float(h.conv_bp.mean()), "dots": h.conv_bp.round(2).tolist()})
    save_svg(strip_plot(rows, title="首发事件：每一次的折算收益率变化", unit="bp", nd=1,
                        desc="每个点是一次首发事件，竖线是均值。", axis_note="折算的收益率变化，bp"), "rpt3-first")

    h = gt[gt["product"] == "T"].set_index("group").reindex(GROUPS)
    panels = [("T 期货折算", h.conv_bp.round(1).tolist(), [""] * 5), ("10 年现券", h.spot_bp.round(1).tolist(), [""] * 5)]
    save_svg(hbar_panels(GROUPS, panels, title="10 年：期货折算和现券的平均变化", unit="bp", nd=1, tick_labels=False,
                         desc="宽松方向，bp；同一批事件窗口。"), "rpt3-spot")

    h = cv[cv.spread == "10年−2年"].set_index("group").reindex(GROUPS)
    panels = [("期货折算（T 对 TS）", h.conv_bp.round(1).tolist(), [""] * 5), ("现券", h.spot_bp.round(1).tolist(), [""] * 5)]
    save_svg(hbar_panels(GROUPS, panels, title="10 年减 2 年利差的变化", unit="bp", nd=1, tick_labels=False,
                         desc="宽松方向，bp；正数是变陡。"), "rpt3-curve")


def main():
    fut = load_futures()
    yields = load("cn_us_yields", "日期")
    events = load("policy_events")
    products = {p: Product(fut, yields, p) for p in BOND}
    ev_pos = {p: event_positions(products[p].cal, events.index) for p in BOND}

    ev = build_events(products, events)
    save_csv(ev, REPORT, "events")
    gt = group_table(ev, products, ev_pos)
    save_csv(gt, REPORT, "groups")
    sp = vs_spot(ev)
    save_csv(sp, REPORT, "vs_spot")
    per, cv = curve(ev)
    save_csv(per, REPORT, "curve_events")
    save_csv(cv, REPORT, "curve")
    dper, dsum = drift(ev, products["T"], ev_pos["T"])
    save_csv(dper, REPORT, "drift_events")
    save_csv(dsum, REPORT, "drift")
    case = case_1030(products)
    save_csv(case, REPORT, "case_20261030")
    dur = pd.DataFrame({p: pd.Series([products[p].duration(d) * 100 for d in products[p].cal[::20]],
                                     index=products[p].cal[::20]) for p in BOND})
    save_csv(dur.reset_index(names="date"), REPORT, "duration_sample")
    charts(gt, cv, ev, sp)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 300):
        print(ev[~ev.usable])
        for name, df in [("groups", gt), ("vs_spot", sp), ("curve", cv), ("drift", dsum), ("drift_events", dper), ("case", case)]:
            print(f"\n== {name}\n", df.round(3))
        print(dur.describe().round(2))


if __name__ == "__main__":
    main()
