import numpy as np
import pandas as pd

from macrob.rpt_bond_futures import Product, event_positions, placebo_pool


def fake_product(n=300, dur=0.08, seed=0):
    """价格 = 100 × (1 − dur × 累计 Δy)，dur 的单位是 %/bp，即 8 年久期。"""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2024-01-01", periods=n)
    dy = rng.normal(scale=2.0, size=n)
    dy[0] = 0
    price = 100 * np.cumprod(1 - dur / 100 * dy)
    p = Product.__new__(Product)
    p.name, p.cal = "T", cal
    p.dom = pd.Series("T2412", index=cal)
    p.px = pd.DataFrame({"T2412": price}, index=cal)
    p.ret = pd.Series(np.r_[np.nan, price[1:] / price[:-1] * 100 - 100], index=cal)
    p.dy = pd.Series(np.r_[np.nan, dy[1:]], index=cal)
    p.y = pd.Series(2 + np.cumsum(dy) / 100, index=cal)
    return p


def test_duration_recovered_and_min_obs():
    p = fake_product()
    assert abs(p.duration(p.cal[-1]) - 0.08) < 1e-3          # 斜率取相反数，单位 %/bp
    assert np.isnan(p.duration(p.cal[30]))                    # 不足 60 天不算


def test_window_spans_announcement_day():
    p = fake_product()
    s, e = p.window(pd.Timestamp("2024-01-10"))               # 周三，交易日
    assert (s, e) == (pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-11"))
    s, e = p.window(pd.Timestamp("2024-01-13"))               # 周六
    assert (s, e) == (pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-15"))


def test_converted_bp_matches_yield_change():
    p = fake_product()
    s, e = p.cal[200], p.cal[202]
    chg = p.change(s, e)
    conv = -chg / p.duration(s)
    assert abs(conv - (p.y[e] - p.y[s]) * 100) < 0.2


def test_placebo_pool_avoids_events():
    p = fake_product(n=100)
    pos = event_positions(p.cal, [p.cal[50]])
    pool_all = placebo_pool(p, 2, np.array([], dtype=int))
    pool = placebo_pool(p, 2, pos)
    # 起点 i 的窗口 [i−10, i+12] 不能碰到第 50 天：i ∈ [38, 60] 共 23 个被去掉
    assert len(pool_all) - len(pool) == 23


def test_event_positions_non_trading_day():
    cal = pd.bdate_range("2024-01-01", periods=10)
    assert list(event_positions(cal, [pd.Timestamp("2024-01-06")])) == [5]   # 周六 → 下周一


def test_duration_neutral_steepener_zero_on_parallel_shift():
    ds, dl = 0.02, 0.08                                       # %/bp
    dy = -5.0
    ret_s, ret_l = -ds * dy, -dl * dy
    assert abs(ret_s - ds / dl * ret_l) < 1e-12
