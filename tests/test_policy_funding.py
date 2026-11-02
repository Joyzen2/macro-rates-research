"""第 18 步的关键规则：资金利率的窗口不算公告当天，回归能还原已知的系数。"""
import numpy as np
import pandas as pd

from macrob import policy_funding as P


def test_around_skips_the_announcement_day_and_needs_full_windows():
    days = pd.bdate_range("2024-01-01", periods=30)
    s = pd.Series(1.0, index=days)
    d = days[15]
    s.loc[d] = 99.0                                   # 公告当天的尖峰不能进窗口
    s.loc[days[16:]] = 1.1
    assert P.around(s, d, 5) == 10.0 or abs(P.around(s, d, 5) - 10.0) < 1e-9
    assert np.isnan(P.around(s, days[2], 5))          # 公告前不够 5 天
    assert np.isnan(P.around(s, days[28], 5))         # 公告后不够 5 天


def test_regress_recovers_coefficients_and_tests_equality():
    rng = np.random.default_rng(0)
    k = pd.DataFrame({"意外(bp)": rng.normal(0, 8, 200), "预期之内(bp)": rng.normal(-5, 8, 200)})
    y = 0.6 * k["意外(bp)"] + 0.0 * k["预期之内(bp)"] + rng.normal(0, 0.5, 200)
    r = P.regress(y, k)
    assert abs(r["意外系数"] - 0.6) < 0.02 and abs(r["预期之内系数"]) < 0.02
    assert r["相等检验p"] < 0.001 and r["事件数"] == 200
