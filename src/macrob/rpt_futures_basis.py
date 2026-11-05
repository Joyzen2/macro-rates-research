"""第一篇：股指期货基差。贴水有多深，多少来自分红，能不能预测收益，吃贴水赚不赚钱。

uv run python -m macrob.rpt_futures_basis
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rpt_common import (EQUITY, complete_from, INDEX_OF, ann_return, dominant, expiry_table, load, load_futures, max_drawdown,
                    month_ends, nw_ols, price_panel, save_csv, save_svg, shift_placebo, spearman)

REPORT = "index-futures-basis"
COST = 0.0002          # 单边成本
ROLL_TDAYS = 5         # 距到期不超过 5 个交易日就换


def annualized_basis(F, S, days):
    return (np.asarray(F) / np.asarray(S) - 1) * 365 / np.asarray(days) * 100


def dividend_rate(tr_t, tr_T, p_t, p_T):
    """区间分红率 d = (TR_T/TR_t)/(P_T/P_t) − 1。"""
    return (tr_T / tr_t) / (p_T / p_t) - 1


def build_daily(fut, price, tr) -> pd.DataFrame:
    """每个品种每天主力合约的年化基差、（到期已过时）年化分红率和去分红基差。"""
    out = []
    last_spot = price.index.max()
    for prod in EQUITY:
        idx = INDEX_OF[prod]
        d = dominant(fut, prod, min_days=5)
        d = d[(d.index <= last_spot) & (d.index >= complete_from(fut, prod))]
        d = d.join(price[idx].rename("S"), how="inner")
        d["basis"] = annualized_basis(d["close"], d["S"], d["days"])
        p_t, tr_t = price[idx].reindex(d.index), tr[idx].reindex(d.index)
        p_T, tr_T = price[idx].reindex(d["expiry"]).values, tr[idx].reindex(d["expiry"]).values
        has = (d["expiry"] <= last_spot).values & np.isfinite(p_T) & np.isfinite(tr_T)
        div = dividend_rate(tr_t.values, tr_T, p_t.values, p_T)
        d["div_ann"] = np.where(has, div * 365 / d["days"] * 100, np.nan)
        d["exdiv_basis"] = d["basis"] + d["div_ann"]
        d["product"] = prod
        out.append(d.reset_index()[["date", "product", "contract", "expiry", "days", "close", "S", "hold", "basis",
                                    "div_ann", "exdiv_basis"]])
    return pd.concat(out, ignore_index=True)


def levels(daily):
    rows, yrows = [], []
    for prod, g in daily.groupby("product", sort=False):
        rows.append({"product": prod, "start": g.date.min().date(), "end": g.date.max().date(), "n": len(g),
                     "median": g.basis.median(), "p10": g.basis.quantile(.1), "p90": g.basis.quantile(.9),
                     "mean": g.basis.mean(), "latest": g.basis.iloc[-1]})
        for y, h in g.groupby(g.date.dt.year):
            yrows.append({"product": prod, "year": y, "n": len(h), "median": h.basis.median(),
                          "p10": h.basis.quantile(.1), "p90": h.basis.quantile(.9)})
    return pd.DataFrame(rows), pd.DataFrame(yrows)


def seasonal(daily):
    daily = daily.assign(month=daily.date.dt.month)
    rows = []
    for (prod, m), g in daily.groupby(["product", "month"], sort=False):
        s = g[g.exdiv_basis.notna()]
        rows.append({"product": prod, "month": m, "n": len(g), "basis_mean": g.basis.mean(),
                     "n_div": len(s), "basis_mean_same": s.basis.mean(), "div_part": -s.div_ann.mean(),
                     "exdiv_mean": s.exdiv_basis.mean()})
    return pd.DataFrame(rows).sort_values(["product", "month"], key=lambda c: c.map({p: i for i, p in enumerate(EQUITY)}) if c.name == "product" else c)


def seasonal_summary(season):
    """看完结果后加的汇总：5–8 月的平均减去其余月份的平均（月份等权）。"""
    rows = []
    for prod, g in season.groupby("product", sort=False):
        hot = g.month.between(5, 8)
        rows.append({"product": prod,
                     "basis_may_aug": g.basis_mean_same[hot].mean(), "basis_other": g.basis_mean_same[~hot].mean(),
                     "basis_gap": g.basis_mean_same[hot].mean() - g.basis_mean_same[~hot].mean(),
                     "exdiv_may_aug": g.exdiv_mean[hot].mean(), "exdiv_other": g.exdiv_mean[~hot].mean(),
                     "exdiv_gap": g.exdiv_mean[hot].mean() - g.exdiv_mean[~hot].mean()})
    return pd.DataFrame(rows)


def decomposition(daily, shibor3m=None):
    rows = []
    for prod, g in daily.groupby("product", sort=False):
        s = g[g.exdiv_basis.notna()]
        rows.append({"product": prod, "n": len(s), "start": s.date.min().date(), "end": s.date.max().date(),
                     "basis": s.basis.mean(), "div_part": -s.div_ann.mean(), "exdiv_part": s.exdiv_basis.mean(),
                     "exdiv_median": s.exdiv_basis.median(), "basis_all": g.basis.mean(),
                     # 看完结果后加的：同一批日子 3 个月 SHIBOR 的平均，持有成本定价下去分红基差应接近它
                     "shibor3m_mean": shibor3m.reindex(s.date).mean() if shibor3m is not None else np.nan})
    return pd.DataFrame(rows)


def monthly_signals(fut, daily, price, tr):
    """每月最后一个交易日：主力基差 → 下个月 (a) 现货全收益，(b) 期货多头超额。"""
    rows = []
    exp_all = expiry_table(fut)
    for prod in EQUITY:
        idx = INDEX_OF[prod]
        g = daily[daily["product"] == prod].set_index("date")
        px = price_panel(fut, prod)
        exp = exp_all[exp_all["product"] == prod].set_index("contract")["expiry"]
        hold = fut[fut["product"] == prod].pivot(index="date", columns="contract", values="hold")
        mes = month_ends(g.index)
        for t0, t1 in zip(mes[:-1], mes[1:]):
            row = g.loc[t0]
            ra = tr[idx][t1] / tr[idx][t0] - 1
            c, sig_b, switched = row.contract, row.basis, False
            if exp[c] < t1:                                  # 下月底之前到期：换成下月底仍存续的主力
                alive = [k for k in hold.columns if k in exp and exp[k] >= t1 and np.isfinite(px.at[t0, k])]
                c = hold.loc[t0, alive].fillna(0).idxmax()
                days = (exp[c] - t0).days
                sig_b = float(annualized_basis(px.at[t0, c], row.S, days))
                switched = True
            rf = px.at[t1, c] / px.at[t0, c] - 1
            rows.append({"product": prod, "month_end": t0.date(), "next_month_end": t1.date(), "contract_a": row.contract,
                         "signal_a": row.basis, "spot_tr_ret": ra * 100, "contract_b": c, "signal_b": sig_b,
                         "switched": switched, "fut_ret": rf * 100, "fut_excess": (rf - ra) * 100})
    return pd.DataFrame(rows)


def predict(sig):
    rows = []
    for prod, g in sig.groupby("product", sort=False):
        for target, sc, yc in [("现货全收益", "signal_a", "spot_tr_ret"), ("期货多头超额", "signal_b", "fut_excess")]:
            x, y = g[sc].values, g[yc].values
            reg = nw_ols(y, x, lags=1)
            rho, p, nsh = shift_placebo(x, y)
            rows.append({"product": prod, "target": target, "n": len(g), "start": g.month_end.min(), "end": g.next_month_end.max(),
                         "spearman": rho, "b": reg["b"], "t_nw": reg["t_b"], "r2": reg["r2"], "placebo_p": p, "n_shifts": nsh})
    return pd.DataFrame(rows)


def trading_days_to(cal: pd.DatetimeIndex, t: pd.Timestamp, expiry: pd.Timestamp) -> int:
    if expiry <= cal[-1]:
        return int(((cal > t) & (cal <= expiry)).sum())
    return int(((cal > t)).sum() + np.busday_count(cal[-1].date() + pd.Timedelta(days=1), expiry.date() + pd.Timedelta(days=1)))


def carry(fut, daily, tr, shibor3m):
    """滚动持有主力多头，全额现金担保，闲置现金按 3 个月 SHIBOR 按日（自然日）计息。"""
    navs, logs = {}, []
    exp_all = expiry_table(fut)
    for prod in EQUITY:
        g = daily[daily["product"] == prod].set_index("date")
        cal = g.index
        px = price_panel(fut, prod).reindex(cal)
        hold = fut[fut["product"] == prod].pivot(index="date", columns="contract", values="hold").reindex(cal)
        exp = exp_all[exp_all["product"] == prod].set_index("contract")["expiry"]
        rate = shibor3m.reindex(cal).ffill()
        cur = g.contract.iloc[0]
        nav = [1.0 - COST]
        n_roll = 0
        for i in range(1, len(cal)):
            t, tp = cal[i], cal[i - 1]
            r = px.at[t, cur] / px.at[tp, cur] - 1
            cash = rate[tp] / 100 * (t - tp).days / 365
            v = nav[-1] * (1 + r + cash)
            dom = g.contract.iloc[i]
            near = trading_days_to(cal, t, exp[cur]) <= ROLL_TDAYS
            new = dom
            if trading_days_to(cal, t, exp[dom]) <= ROLL_TDAYS:   # 主力自己也快到期：不能再持有它，选更远的合约里持仓最大的，否则换走第二天又换回来
                later = [k for k in hold.columns if exp.get(k, t) > exp[dom] and np.isfinite(px.at[t, k])]
                new = hold.loc[t, later].fillna(0).idxmax()
            if new != cur:
                v *= (1 - 2 * COST)
                logs.append({"product": prod, "date": t.date(), "from": cur, "to": new, "reason": "临近到期" if near else "主力变了"})
                cur, n_roll = new, n_roll + 1
            nav.append(v)
        nav = pd.Series(nav, index=cal)
        bench = tr[INDEX_OF[prod]].reindex(cal)
        navs[prod] = pd.DataFrame({"strategy": nav, "tr_index": bench / bench.iloc[0]})
        navs[prod].attrs["n_roll"] = n_roll
    return navs, pd.DataFrame(logs)


def carry_tables(navs):
    rows, yrows = [], []
    for prod, df in navs.items():
        s, b = df.strategy, df.tr_index
        rs, rb = s.pct_change().dropna(), b.pct_change().dropna()
        rows.append({"product": prod, "start": df.index[0].date(), "end": df.index[-1].date(), "n_roll": df.attrs["n_roll"],
                     "ann_ret": ann_return(s) * 100, "ann_vol": rs.std() * np.sqrt(244) * 100, "mdd": max_drawdown(s) * 100,
                     "tr_ann_ret": ann_return(b) * 100, "tr_ann_vol": rb.std() * np.sqrt(244) * 100, "tr_mdd": max_drawdown(b) * 100,
                     "ann_excess": (ann_return(s) - ann_return(b)) * 100,
                     "te": (rs - rb).std() * np.sqrt(244) * 100})
        for y, h in df.groupby(df.index.year):
            prev = df[df.index.year < y]
            base = prev.iloc[-1] if len(prev) else h.iloc[0]
            ys, yb = h.strategy.iloc[-1] / base.strategy - 1, h.tr_index.iloc[-1] / base.tr_index - 1
            yrows.append({"product": prod, "year": y, "partial": bool(len(prev) == 0 or y == df.index[-1].year),
                          "strategy": ys * 100, "tr_index": yb * 100, "excess": (ys - yb) * 100})
    return pd.DataFrame(rows), pd.DataFrame(yrows)


def charts(daily, season, decomp, carry_y):
    from charts import S, daily_lines, hbar_panels, stacked_hbar, hbar, DIM, INK, NEG
    wide = daily.pivot(index="date", columns="product", values="basis")[EQUITY]
    ma = wide.rolling(20, min_periods=15).mean()
    ma = ma[ma.index >= "2019-01-01"]
    series = [(p, ma[p].round(2).tolist(), S[i], 1.3) for i, p in enumerate(EQUITY)]
    svg = daily_lines(list(ma.index), series, title="四个品种主力合约的年化基差（20 日均线）", ylo=-32, yhi=15,
                      yticks=[-30, -20, -10, 0, 10], unit="%", nd=1, month_step=12, ylabel="%",
                      desc="2019 年 4 月至 2026 年 10 月，年化基差的 20 日均线，负数是贴水。")
    save_svg(svg, "rpt2-basis")

    months = [f"{m}月" for m in range(1, 13)]
    panels = []
    for prod in ["IF", "IC"]:
        sm_ = season[season["product"] == prod].set_index("month").reindex(range(1, 13))
        panels.append((f"{prod} 基差", sm_.basis_mean_same.round(1).tolist(), [""] * 12))
        panels.append((f"{prod} 去分红", sm_.exdiv_mean.round(1).tolist(), [""] * 12))
    save_svg(hbar_panels(months, panels, title="按月份的平均年化基差：去掉分红前后", unit="%", nd=1, tick_labels=False,
                         desc="只用到期日已过的合约，同一批交易日；年化，%。IH 和 IF 相似，IM 和 IC 相似，见结果表。"), "rpt2-season")

    m = [[r.div_part, r.exdiv_part] for r in decomp.itertuples()]
    save_svg(stacked_hbar(list(decomp["product"]), ["分红部分", "去分红部分"], m, title="年化基差的来源", unit="%", nd=1,
                          desc="全样本平均，只用到期日已过的合约。", axis_note="年化，%"), "rpt2-decomp")

    full = carry_y[~carry_y.partial]
    rows = []
    for prod in EQUITY:
        h = full[full["product"] == prod]
        rows.append({"label": prod, "sub": f"{h.year.min()}–{h.year.max()} 年", "value": float(h.excess.mean())})
    save_svg(hbar(rows, title="吃贴水：完整年份相对全收益指数的平均超额", unit="%", nd=1, diverging=True,
                  desc="每年策略收益减全收益指数收益，再对完整年份取平均。", axis_note="超额收益，%"), "rpt2-carry")


def main():
    fut = load_futures()
    price, tr = load("index_price"), load("index_total_return")
    shibor3m = load("funding_daily")["SHIBOR_3M"].dropna()

    daily = build_daily(fut, price, tr)
    save_csv(daily, REPORT, "daily")
    lv, lvy = levels(daily)
    save_csv(lv, REPORT, "levels")
    save_csv(lvy, REPORT, "levels_by_year")
    season = seasonal(daily)
    save_csv(season, REPORT, "seasonal")
    save_csv(seasonal_summary(season), REPORT, "seasonal_summary")
    decomp = decomposition(daily, shibor3m)
    save_csv(decomp, REPORT, "decomposition")
    sig = monthly_signals(fut, daily, price, tr)
    save_csv(sig, REPORT, "monthly_signals")
    pr = predict(sig)
    save_csv(pr, REPORT, "predict")
    navs, rolls = carry(fut, daily, tr, shibor3m)
    save_csv(rolls, REPORT, "carry_rolls")
    cs, cy = carry_tables(navs)
    save_csv(cs, REPORT, "carry")
    save_csv(cy, REPORT, "carry_by_year")
    nav_all = pd.concat({p: d for p, d in navs.items()}, names=["product", "date"]).reset_index()
    save_csv(nav_all, REPORT, "carry_nav")
    charts(daily, season, decomp, cy)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 200):
        for name, df in [("levels", lv), ("levels_by_year", lvy), ("seasonal", season), ("seasonal_summary", seasonal_summary(season)), ("decomposition", decomp),
                         ("predict", pr), ("carry", cs), ("carry_by_year", cy)]:
            print(f"\n== {name}\n", df.round(3))


if __name__ == "__main__":
    main()
