"""重建项目的关键规则：发布滞后、稳健标准化、实时 PCA、风险预算映射、信贷脉冲的计算。用人造数据测，不联网。"""
import numpy as np
import pandas as pd

from macrob import allocation, datalayer, state


def _months(n, start="2015-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def test_realtime_panel_respects_publication_lag(monkeypatch):
    idx = _months(6)
    raw = pd.DataFrame({"pmi": np.arange(6.0), "ip_yoy": np.arange(6.0) * 10}, index=idx)
    monkeypatch.setattr(datalayer, "raw_panel", lambda: raw)
    rt = datalayer.realtime_panel()
    t = idx[3]
    assert rt.loc[t, "pmi"] == 2.0            # 2015 年的 PMI 在次月 1 日发布：4 月末只知道 3 月的
    assert rt.loc[t, "ip_yoy"] == 20.0         # 滞后 1：月末 t 只知道 t−1 月的数


def test_pmi_lag_uses_actual_release_dates():
    lag = datalayer.pmi_lag()
    assert lag[pd.Timestamp("2016-11-30")] == 1      # 2016-12-01 才发布
    assert lag[pd.Timestamp("2018-01-31")] == 0      # 当月最后一天（周三）发布，收盘前就有
    assert lag[pd.Timestamp("2018-03-31")] == 1      # 最后一天是周六，最后一个交易日已经收盘


def test_robust_z_uses_only_the_past_and_is_clipped():
    s = pd.Series(np.r_[np.random.default_rng(0).normal(size=40), 100.0], index=_months(41))
    z = state.robust_z(s)
    assert z.iloc[:state.MIN_HIST - 1].isna().all()
    assert z.iloc[-1] == state.CLIP                          # 极端值截到 +3
    z_trunc = state.robust_z(s.iloc[:40])
    assert np.allclose(z.iloc[:40].dropna(), z_trunc.dropna())    # 加一个未来的观测，不改变过去的值


def test_floor_budgets_keep_every_asset_positive_and_sum_to_one():
    b = allocation.floor_budgets([1.0, 0.0, 0.0])
    assert np.isclose(b.sum(), 1) and (b >= allocation.FLOOR).all()


def test_pca_factor_uses_only_past_data_and_keeps_direction():
    keys = [i.key for i in datalayer.REGISTRY if i.category == "生产"][:4]
    rng = np.random.default_rng(0)
    common = 2 * np.sin(np.arange(80) / 6)                               # 有周期的平稳共同因子
    z = pd.DataFrame({k: common + rng.normal(scale=0.5, size=80) for k in keys}, index=_months(80))
    full = state.pca_category_factors(z)["生产"]
    cut = state.pca_category_factors(z.iloc[:60])["生产"]
    assert np.isclose(full.iloc[59], cut.iloc[59])                        # 截掉后面的数据，前面的值不变
    assert full.iloc[:35].isna().all() and full.iloc[35:].notna().all()   # 至少 36 个月才开始
    assert np.corrcoef(full.dropna(), common[35:])[0, 1] > 0.8            # 共同因子上升时主成分也上升


def test_risk_budget_survives_tiny_variances():
    from macrob import riskbudget as rb
    S = np.diag([3.013896e-03, 1.384197e-03, 3.953956e-05])      # 2020 年附近真实出现过、CLARABEL 直接解不动的一组
    b = np.array([0.168844, 0.02, 0.811156])
    w = np.sqrt(b / np.diag(S))
    assert np.allclose(rb.risk_budget_weights(S, b), w / w.sum(), atol=1e-4)  # 对角协方差有解析解


def test_regime_probabilities_are_reproducible():
    from macrob import regime
    rng = np.random.default_rng(1)
    y = pd.Series(np.r_[rng.normal(0.5, 0.3, 40), rng.normal(-0.5, 0.6, 40), rng.normal(0.5, 0.3, 40)], index=_months(120))
    a, b = regime.realtime(y), regime.realtime(y)
    assert len(a) > 0 and a.equals(b)                                    # 同样的数据，每次结果都一样
