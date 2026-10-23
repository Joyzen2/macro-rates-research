"""第 2、3 步：特征、类别因子、增长和通胀两个维度、四象限概率。运行：uv run python -m macrob.state

规则（看结果之前定的）：
- 特征：每个指标在实时面板上做扩张窗口的稳健标准化：(当月值 − 到当月为止的中位数) ÷ (1.4826 × MAD)，至少 36 个月，截到 ±3。
- 类别因子（第 14 步起的主方案）：每类第一主成分，每个月末只用到当月为止的数据重新估计（扩张窗口）。
  对照：5 年滚动窗口的 PCA；该类指标标准化值的等权平均（第 14 步之前的主方案）。
  方向：越高 = 越强 / 越热；货币类越高 = 越紧（期限利差取反）。
- 增长维度 = 生产、投资、进出口三类的平均（和实习时一样）；通胀维度 = 通胀类。稳健性：增长维度加上消费类。
- 状态：两个维度的 3 个月变化各跑一个两状态区制模型，取实时滤波的「加速」概率；四象限概率 = 两个概率相乘。
- 拐点评估：参考拐点、配对窗口（±6 个月）、安慰剂和第一轮完全一样。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cycles, evaluate, regime
from .datalayer import CATEGORIES, REGISTRY, RESULTS, ROOT, raw_panel, realtime_panel, with_csmar_history

MIN_HIST, CLIP = 36, 3.0
SIGN = {"term": -1}                                    # 期限利差越陡通常越松，所以在「货币越紧」里取反
GROWTH = ["生产", "投资", "进出口"]


def robust_z(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index)
    v = s.to_numpy()
    for i in range(len(v)):
        hist = v[: i + 1][~np.isnan(v[: i + 1])]
        if len(hist) < MIN_HIST or np.isnan(v[i]):
            continue
        med = np.median(hist)
        mad = 1.4826 * np.median(np.abs(hist - med))
        if mad > 0:
            out.iloc[i] = np.clip((v[i] - med) / mad, -CLIP, CLIP)
    return out


def features() -> pd.DataFrame:
    f = ROOT / "data" / "features.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rt = realtime_panel(with_csmar_history(raw_panel()))          # 第 14 步：五个指标补了 2005–2007 年的历史
    z = pd.DataFrame({k: robust_z(rt[k]) * SIGN.get(k, 1) for k in rt.columns})
    z.to_parquet(f)
    return z


def ew_category_factors(z: pd.DataFrame | None = None) -> pd.DataFrame:
    """对照：该类指标标准化值的等权平均（至少 2 个指标有值）。"""
    z = features() if z is None else z
    cat = {i.key: i.category for i in REGISTRY}
    out = {}
    for c in CATEGORIES:
        cols = [k for k in z.columns if cat.get(k) == c]
        x = z[cols]
        out[c] = x.mean(axis=1).where(x.notna().sum(axis=1) >= 2)
    return pd.DataFrame(out)


def pca_category_factors(z: pd.DataFrame | None = None, window: int | None = None, info: dict | None = None) -> pd.DataFrame:
    """每类第一主成分。每个月末只用到当月为止的完整样本（至少 36 个月、至少 2 个指标），
    列先按样本内均值、标准差标准化；符号定为载荷之和为正（所有特征都已是「越高越强 / 越热」）。
    window=None 是扩张窗口；window=60 只用最近 60 个月。传入 info 时记下每个月的解释方差和载荷。"""
    z = features() if z is None else z
    cat = {i.key: i.category for i in REGISTRY}
    out = {}
    for c in CATEGORIES:
        x = z[[k for k in z.columns if cat.get(k) == c]]
        vals = pd.Series(np.nan, index=z.index)
        for i, t in enumerate(z.index):
            h = x.iloc[: i + 1]
            cols = [k for k in h.columns if h[k].notna().sum() >= MIN_HIST and pd.notna(h[k].iloc[-1])]
            if len(cols) < 2:
                continue
            h = h[cols].dropna()
            if window:
                h = h.iloc[-window:]
            if len(h) < MIN_HIST:
                continue
            mu, sd = h.mean(), h.std().replace(0, np.nan)
            hs = ((h - mu) / sd).dropna(axis=1)
            if hs.shape[1] < 2:
                continue
            w, v = np.linalg.eigh(np.cov(hs.to_numpy().T))
            load = v[:, -1] * (1 if v[:, -1].sum() >= 0 else -1)
            vals[t] = float(((hs.iloc[-1] * load).sum()) / np.sqrt(w[-1]))
            if info is not None:
                info.setdefault(c, []).append({"月份": t, "解释方差": float(w[-1] / w.sum()), **dict(zip(hs.columns, load))})
        out[c] = vals
    return pd.DataFrame(out)


def category_factors(z: pd.DataFrame | None = None) -> pd.DataFrame:
    """主方案（第 14 步）：扩张窗口 PCA。"""
    return pca_category_factors(z)


def dimensions(cf: pd.DataFrame | None = None, growth_cats=GROWTH) -> pd.DataFrame:
    cf = category_factors() if cf is None else cf
    return pd.DataFrame({"增长": cf[growth_cats].mean(axis=1, skipna=False), "通胀": cf["通胀"]}).dropna()


def quadrant_probs(dims: pd.DataFrame) -> pd.DataFrame:
    pg = regime.realtime((dims["增长"] - dims["增长"].shift(3)).dropna())
    pi = regime.realtime((dims["通胀"] - dims["通胀"].shift(3)).dropna())
    q = pd.concat([pg.rename("增长加速"), pi.rename("通胀加速")], axis=1).dropna()
    q["复苏"] = q["增长加速"] * (1 - q["通胀加速"])
    q["过热"] = q["增长加速"] * q["通胀加速"]
    q["滞胀"] = (1 - q["增长加速"]) * q["通胀加速"]
    q["衰退"] = (1 - q["增长加速"]) * (1 - q["通胀加速"])
    return q


def turning_point_table(signals: dict[str, pd.Series], start, end, confirm: bool = False) -> pd.DataFrame:
    refs = {"PMI": cycles.turning_points(cycles.pmi()), "工业增加值同比": cycles.turning_points(cycles.industrial_yoy())}
    months = pd.date_range("2000-01-31", "2030-12-31", freq="ME")
    rng = np.random.default_rng(0)
    rows = []
    for rname, ref in refs.items():
        for name, p in signals.items():
            sw = regime.switches(p.dropna(), confirm=confirm)
            sc = evaluate.score(ref, sw, start, end)
            win = months[(months >= start) & (months <= end)]
            n_sw = max(sc["模型切换数"], 1)
            fake = []
            for _ in range(2000):
                d = np.sort(rng.choice(win, size=min(n_sw, len(win)), replace=False))
                k0 = rng.integers(2)
                fake.append(evaluate.score(ref, pd.DataFrame({"date": pd.to_datetime(d), "kind": ["peak" if (k0 + i) % 2 == 0 else "trough" for i in range(len(d))]}), start, end)["捕捉率"])
            fake = np.array(fake)
            rows.append({"参考": rname, "版本": name, **{k: v for k, v in sc.items() if k != "明细"},
                         "瞎猜捕捉率": fake.mean(), "瞎猜不差于它的概率": float((fake >= sc["捕捉率"]).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 240)
    info = {}
    cf = pca_category_factors(info=info)
    ew, roll = ew_category_factors(), pca_category_factors(window=60)
    cf.to_csv(RESULTS / "category_factors.csv", encoding="utf-8-sig")
    ew.to_csv(RESULTS / "category_factors_等权.csv", encoding="utf-8-sig")
    roll.to_csv(RESULTS / "category_factors_滚动.csv", encoding="utf-8-sig")
    # PCA 的诊断：解释方差、最新一期的载荷、和等权版的相关
    names = {i.key: i.name for i in REGISTRY}
    diag, loads = [], []
    for c in CATEGORIES:
        d = pd.DataFrame(info[c]).set_index("月份")
        ev = d["解释方差"]
        last = d.drop(columns="解释方差").iloc[-1].dropna()
        diag.append({"类别": c, "起": f"{ev.index.min():%Y-%m}", "最新指标数": len(last), "解释方差中位数": ev.median(),
                     "解释方差最小": ev.min(), "解释方差最大": ev.max(), "和等权版的相关": float(cf[c].corr(ew[c])),
                     "和滚动窗口版的相关": float(cf[c].corr(roll[c]))})
        loads += [{"类别": c, "指标": names.get(k, k), "载荷": float(v)} for k, v in last.items()]
    diag = pd.DataFrame(diag)
    diag.to_csv(RESULTS / "pca_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(loads).to_csv(RESULTS / "pca_loadings_latest.csv", index=False, encoding="utf-8-sig")
    print("PCA 诊断：\n", diag.round(3).to_string(index=False))
    dims, dims_ew, dims_roll = dimensions(cf), dimensions(ew), dimensions(roll)
    dims_c = dimensions(cf, GROWTH + ["消费"])
    print("类别因子覆盖：", {c: f"{cf[c].first_valid_index():%Y-%m}" for c in cf.columns}, "| 两个维度", dims.index.min().date(), "~", dims.index.max().date())
    print("增长维度和等权版、滚动窗口版的相关：", round(float(pd.concat([dims["增长"], dims_ew["增长"]], axis=1).dropna().corr().iloc[0, 1]), 3),
          round(float(pd.concat([dims["增长"], dims_roll["增长"]], axis=1).dropna().corr().iloc[0, 1]), 3))
    q, q_ew, q_roll = quadrant_probs(dims), quadrant_probs(dims_ew), quadrant_probs(dims_roll)
    q.to_csv(RESULTS / "quadrant_probabilities.csv", encoding="utf-8-sig")
    q_ew.to_csv(RESULTS / "quadrant_probabilities_等权.csv", encoding="utf-8-sig")
    q_roll.to_csv(RESULTS / "quadrant_probabilities_滚动.csv", encoding="utf-8-sig")
    # 7 类各自的景气后验：只做展示，不做检验
    post = pd.DataFrame({c: regime.realtime((cf[c] - cf[c].shift(3)).dropna()) for c in CATEGORIES})
    post.to_csv(RESULTS / "category_posteriors.csv", encoding="utf-8-sig")
    print("7 类景气后验（最近 3 个月）：\n", post.tail(3).round(2).to_string())
    # 拐点：三个版本用同一个评估区间，切换记在确认月
    sig = {"主方案·PCA 扩张窗口": q["增长加速"], "对照·PCA 5 年滚动窗口": q_roll["增长加速"], "对照·等权": q_ew["增长加速"]}
    start = max(v.dropna().index.min() for v in sig.values()) + pd.DateOffset(months=6)
    end = min(v.dropna().index.max() for v in sig.values()) - pd.DateOffset(months=6)
    t = turning_point_table(sig, start, end, confirm=True).assign(评估区间=f"{start:%Y-%m} ~ {end:%Y-%m}")
    # 稳健性：增长维度加上消费类。消费类 2013 年才有 PCA，评估区间单独算
    pc = regime.realtime((dims_c["增长"] - dims_c["增长"].shift(3)).dropna())
    s2, e2 = pc.index.min() + pd.DateOffset(months=6), pc.index.max() - pd.DateOffset(months=6)
    t = pd.concat([t, turning_point_table({"稳健性·加上消费类": pc}, s2, e2, confirm=True).assign(评估区间=f"{s2:%Y-%m} ~ {e2:%Y-%m}")], ignore_index=True)
    t.to_csv(RESULTS / "turning_point_scores.csv", index=False, encoding="utf-8-sig")
    print(f"拐点评估区间 {start.date()} ~ {end.date()}"); print(t.round(3).to_string(index=False))
    t2 = turning_point_table({"主方案·PCA 扩张窗口": q["增长加速"]}, pd.Timestamp("2015-07-31"), pd.Timestamp("2024-12-31"), confirm=True)
    t2.to_csv(RESULTS / "turning_point_scores_round1_window.csv", index=False, encoding="utf-8-sig")
    print("和第一轮同一区间（2015-07 至 2024-12）："); print(t2.round(3).to_string(index=False))
    print("最近 6 个月的四象限概率：\n", q[["复苏", "过热", "滞胀", "衰退"]].tail(6).round(2).to_string())

if __name__ == "__main__":
    main()
