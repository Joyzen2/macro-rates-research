import numpy as np
import pandas as pd

from macrob.rpt_common import (complete_from, contract_month, dominant, expiry_table, nth_friday, nw_ols, random_window_placebo,
                    rule_expiry, shift_placebo)


def test_nth_friday():
    assert str(nth_friday(2026, 12, 3)) == "2026-12-18"     # 股指：第三个周五
    assert str(nth_friday(2026, 12, 2)) == "2026-12-11"     # 国债：第二个周五
    assert str(nth_friday(2026, 5, 3)) == "2026-05-15"


def test_contract_month():
    assert contract_month("IC2612") == (2026, 12)
    assert contract_month("TL2703") == (2027, 3)


def test_rule_expiry_holiday_rolls_forward():
    cal = pd.DatetimeIndex(["2026-10-15", "2026-10-19", "2026-10-20"])   # 10-16 不是交易日
    assert rule_expiry("IF2610", "IF", cal) == pd.Timestamp("2026-10-19")
    assert rule_expiry("IF2612", "IF", cal) == pd.Timestamp("2026-12-18")   # 超出日历范围：按规则


def _toy():
    rows = []
    for d, h1, h2 in [("2026-01-05", 100, 50), ("2026-01-12", 100, 50), ("2026-01-13", 100, 50), ("2026-01-20", 0, 10)]:
        rows.append({"date": pd.Timestamp(d), "close": 1.0, "hold": h1, "contract": "IF2601", "product": "IF"})
        rows.append({"date": pd.Timestamp(d), "close": 1.0, "hold": h2, "contract": "IF2602", "product": "IF"})
    rows.append({"date": pd.Timestamp("2026-01-21"), "close": 1.0, "hold": 10, "contract": "IF2602", "product": "IF"})
    rows.append({"date": pd.Timestamp("2026-01-21"), "close": 1.0, "hold": 5, "contract": "IF2603", "product": "IF"})
    return pd.DataFrame(rows)


def test_expiry_uses_last_date_for_expired_contracts():
    e = expiry_table(_toy()).set_index("contract")
    assert e.loc["IF2601", "expiry"] == pd.Timestamp("2026-01-20")     # 已到期：数据里的最后一天
    assert e.loc["IF2602", "expiry"] == pd.Timestamp("2026-02-20")     # 还在交易：第三个周五


def test_dominant_min_days():
    d = dominant(_toy(), "IF", min_days=5)
    assert d.loc["2026-01-12", "contract"] == "IF2601"   # 剩 8 天，持仓最大
    assert d.loc["2026-01-20", "contract"] == "IF2602"   # IF2601 剩 0 天，排除
    d0 = dominant(_toy(), "IF")
    assert d0.loc["2026-01-13", "contract"] == "IF2601"


def test_complete_from():
    # 01-20 只有一个合约持仓量为正，算缺口；从下一天起用
    assert complete_from(_toy(), "IF") == pd.Timestamp("2026-01-21")


def test_nw_ols_recovers_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = 0.5 + 2 * x + rng.normal(scale=0.1, size=500)
    r = nw_ols(y, x, lags=5)
    assert abs(r["b"] - 2) < 0.02 and r["t_b"] > 50


def test_shift_placebo_counts_offsets():
    x = np.arange(20, dtype=float)
    real, p, n = shift_placebo(x, x)
    assert n == 20 - 12 + 1 and abs(real - 1) < 1e-12 and p <= 1


def test_random_window_placebo_reproducible():
    vals = {1: np.arange(100, dtype=float)}
    a = random_window_placebo(vals, [1, 1, 1], real_mean=99.0)
    b = random_window_placebo(vals, [1, 1, 1], real_mean=99.0)
    assert a == b and a[0] < 0.05
    assert random_window_placebo(vals, [1, 1, 1], real_mean=a[1])[0] > 0.9
