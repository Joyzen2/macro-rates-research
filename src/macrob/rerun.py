"""第 8 步：把第一轮的泰勒规则和断点检验接到重建后的数据上。运行：uv run python -m macrob.rerun

和第一轮的区别：政策利率换成 7 天回购定盘利率 FR007（月均），增长缺口另做一版 PMI − 50；
断点检验用重建后的类别因子（货币类 → 下月 10 年国债总收益，投资类 → 下月沪深300 全收益）。
其余规则和第一轮一样：反应函数扩张窗口估计（至少 60 个月），月末只用上月已公布的数；久期择时扣 0.05% 切换成本；
平移安慰剂 2000 次；断点数 0–3、每段至少 36 个月、BIC 选断点数。
M2 意外：口径和第一轮的意外研究一样（公布值 − 金十预测值，除以之前所有发布的意外标准差，至少 12 次）。
窗口原定「发布日前一个交易日收盘 → 发布日收盘」；看完结果后发现央行金融数据常在周五收盘后或周末公布，
这个窗口会漏掉反应，所以加了修正窗口「发布日前一个交易日收盘 → 发布日之后第一个交易日收盘」，两个都报。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import markets
from . import sources as S
from .allocation import daily_levels
from .datalayer import RESULTS, raw_panel, realtime_panel
from .probe_util import RAW
from .state import category_factors

MIN_FIT, COST, N_PERM, MIN_SEG = 60, 0.0005, 2000, 36


def taylor_inputs(gap: str) -> pd.DataFrame:
    raw = raw_panel()
    i = raw["fr007"].rename("i")
    pi = raw["cpi_yoy"].rename("pi")
    if gap == "工业增加值":
        ip = raw["ip_yoy"].resample("ME").last().interpolate(limit=1)
        x = ip - ip.rolling(36, min_periods=24).mean().shift(1)
    else:
        x = realtime_panel()["pmi"] - 50                       # 月末已经公布的 PMI（2017 年前次月 1 日才发布）
    d = pd.concat([i, pi, x.rename("x")], axis=1)
    d["pi"] = d["pi"].shift(1)                     # 月末只知道上月的 CPI
    if gap == "工业增加值":
        d["x"] = d["x"].shift(1)                   # 工业增加值也是次月公布
    d["i_lag"] = d["i"].shift(1)
    return d.dropna()


def stance(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in range(MIN_FIT, len(d) + 1):
        w = d.iloc[:k]
        r = sm.OLS(w["i"], sm.add_constant(w[["i_lag", "pi", "x"]])).fit()
        a, rho, bp, bx = r.params[["const", "i_lag", "pi", "x"]]
        last = w.iloc[-1]
        i_star = (a + bp * last["pi"] + bx * last["x"]) / (1 - rho) if rho < 0.995 else np.nan
        rows.append({"date": w.index[-1], "gap": last["i"] - i_star, "rho": rho, "b_pi": bp, "b_x": bx})
    return pd.DataFrame(rows).set_index("date")


def duration_timing(sig: pd.Series, bond: pd.Series, cash: pd.Series) -> float:
    pos = (sig > 0).astype(float)
    r_b = bond.pct_change().shift(-1)
    net = (pos * r_b + (1 - pos) * cash - pos.diff().abs().fillna(0) * COST).dropna()
    ex = net - cash.reindex(net.index)
    return float(ex.mean() / ex.std() * np.sqrt(12))


def best_breaks(x: np.ndarray, y: np.ndarray, max_b: int = 3):
    n = len(x)
    def ssr(i, j):
        X = np.column_stack([np.ones(j - i), x[i:j]])
        b, *_ = np.linalg.lstsq(X, y[i:j], rcond=None)
        e = y[i:j] - X @ b
        return float(e @ e)
    f = {0: {j: ssr(0, j) for j in range(MIN_SEG, n + 1)}}
    arg = {}
    for m in range(1, max_b + 1):
        f[m], arg[m] = {}, {}
        for j in range((m + 1) * MIN_SEG, n + 1):
            c = [(f[m - 1][i] + ssr(i, j), i) for i in range(m * MIN_SEG, j - MIN_SEG + 1) if i in f[m - 1]]
            if c:
                f[m][j], arg[m][j] = min(c)
    bic, cuts_all = [], []
    for m in range(max_b + 1):
        if n not in f[m]:
            break
        bic.append(n * np.log(f[m][n] / n) + (3 * m + 2) * np.log(n))
        cuts, j = [], n
        for mm in range(m, 0, -1):
            j = arg[mm][j]
            cuts.append(j)
        cuts_all.append(sorted(cuts))
    m = int(np.argmin(bic))
    return m, cuts_all[m], bic


def m2_surprise() -> pd.DataFrame:
    d = pd.read_csv(RAW / "consensus_m2.csv")
    d = d.assign(date=pd.to_datetime(d["日期"]), a=pd.to_numeric(d["今值"], errors="coerce"),
                 f=pd.to_numeric(d["预测值"], errors="coerce")).dropna(subset=["a", "f"])
    d = d.drop_duplicates("date").sort_values("date")
    raw = d["a"] - d["f"]
    d["s"] = raw / raw.expanding(12).std().shift(1)              # 只用之前的发布
    d = d.dropna(subset=["s"])
    d = d[d["date"] >= "2011-01-01"]
    mk = markets.daily()
    days = mk.index
    rel = d["date"].dt.normalize()
    windows = {"原定窗口（到发布日收盘）": days.searchsorted(rel, side="left"),
               "修正窗口（到发布日之后第一个交易日收盘）": days.searchsorted(rel, side="right")}
    before = days.searchsorted(rel, side="left") - 1             # 发布日前最后一个交易日
    rows = []
    rng = np.random.default_rng(0)
    for wname, after in windows.items():
        ok = (before >= 0) & (after < len(days))
        x = d["s"].to_numpy()[ok]
        a, b = mk.iloc[after[ok]].reset_index(drop=True), mk.iloc[before[ok]].reset_index(drop=True)
        for col, y in (("Δ10年(bp)", (a["y10"] - b["y10"]) * 100), ("Δ2年(bp)", (a["y2"] - b["y2"]) * 100),
                       ("沪深300(%)", (a["hs300"] / b["hs300"] - 1) * 100)):
            y = y.to_numpy()
            r = sm.OLS(y, sm.add_constant(x), missing="drop").fit(cov_type="HC1")
            m = ~np.isnan(y)
            perm = np.array([np.polyfit(rng.permutation(x[m]), y[m], 1)[0] for _ in range(N_PERM)])
            rows.append({"窗口": wname, "资产": col, "每 1 个标准差的反应": round(float(r.params[1]), 3),
                         "t": round(float(r.tvalues[1]), 2), "打乱 p": round(float((np.abs(perm) >= abs(r.params[1])).mean()), 3),
                         "事件数": int(r.nobs)})
    return pd.DataFrame(rows)


def main() -> None:
    lv = daily_levels().resample("ME").last()
    cash = S.shibor3m_daily().resample("ME").mean() / 1200
    y10 = S.yields_daily()["y10"].resample("ME").last()
    rng = np.random.default_rng(0)
    print("== 泰勒规则（政策利率 = FR007）")
    taylor = []
    for gap in ("工业增加值", "PMI"):
        s = stance(taylor_inputs(gap))
        s = s[s.index >= "2012-01-01"]
        fwd = (y10.shift(-1) - y10).reindex(s.index) * 100
        ic = float(s["gap"].corr(fwd, method="spearman"))
        bond = lv["债"].reindex(s.index)
        c = cash.reindex(s.index)
        real = duration_timing(s["gap"], bond, c)
        bh = duration_timing(pd.Series(1.0, index=s.index), bond, c)
        fake = np.array([duration_timing(pd.Series(np.roll(s["gap"].to_numpy(), k), index=s.index), bond, c)
                         for k in rng.integers(12, len(s) - 12, size=N_PERM)])
        last = s.iloc[-1]
        print(f"增长缺口用{gap}：ρ {last['rho']:.3f}，长期通胀反应 {last['b_pi'] / (1 - last['rho']):.2f}、增长 {last['b_x'] / (1 - last['rho']):.2f}；"
              f"松紧和下月 10 年收益率变化的秩相关 {ic:.3f}；择时夏普 {real:.2f}，一直持有 {bh:.2f}，安慰剂不差于它的概率 {(fake >= real).mean():.2f}")
        taylor.append({"增长缺口": gap, "月数": len(s), "利率平滑 ρ": round(last["rho"], 3),
                       "长期通胀反应": round(last["b_pi"] / (1 - last["rho"]), 2), "长期增长反应": round(last["b_x"] / (1 - last["rho"]), 2),
                       "松紧与下月 Δ10年 秩相关": round(ic, 3), "久期择时夏普": round(real, 2), "一直持有夏普": round(bh, 2),
                       "安慰剂不差于它的概率": round(float((fake >= real).mean()), 2)})
    pd.DataFrame(taylor).to_csv(RESULTS / "taylor_rule.csv", index=False, encoding="utf-8-sig")
    print("== 断点检验（重建后的类别因子）")
    cf = category_factors()
    mo = lv.pct_change()
    rows = []
    for name, (cat, asset) in {"货币类 → 下月 10 年国债": ("货币", "债"), "投资类 → 下月沪深300": ("投资", "股")}.items():
        d = pd.concat([cf[cat].rename("x"), mo[asset].shift(-1).rename("y")], axis=1).dropna()
        m, cuts, bic = best_breaks(d["x"].to_numpy(), d["y"].to_numpy())
        ic = d["x"].corr(d["y"], method="spearman")
        rows.append({"关系": name, "月数": len(d), "BIC 选的断点数": m, "断点": [d.index[c].strftime("%Y-%m") for c in cuts], "整段 IC": round(ic, 3)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS / "structural_breaks.csv", index=False, encoding="utf-8-sig")
    print("== M2 一致预期重试")
    try:
        import akshare as ak
        from .probe_util import cached
        d = cached("consensus_m2", ak.macro_china_m2_yearly)
        has = d.dropna(subset=["今值", "预测值"])
        print(f"拿到了：{len(d)} 次发布，{len(has)} 次有预测值，最早 {pd.to_datetime(has['日期']).min().date() if len(has) else '无'}")
    except Exception as e:
        print("还是失败：", str(e)[:100])
        return
    m2 = m2_surprise()
    m2.to_csv(RESULTS / "m2_surprise_event_study.csv", index=False, encoding="utf-8-sig")
    print(m2.to_string(index=False))


if __name__ == "__main__":
    main()
