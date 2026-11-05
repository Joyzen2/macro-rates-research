import numpy as np
import pandas as pd

from macrob.rpt_cn_us_spread import daily_pairs, match_prev_us, weekly_pairs


def test_match_prev_us_is_strictly_earlier():
    cn = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-16"])
    us = pd.DatetimeIndex(["2024-01-02", "2024-01-12"])          # 01-15 美国休市
    m = pd.DatetimeIndex(match_prev_us(cn, us))
    assert pd.isna(m[0])                                          # 同一天的美国收盘在中国收盘之后，不能用
    assert m[1] == pd.Timestamp("2024-01-02")
    assert m[2] == pd.Timestamp("2024-01-12")


def _toy():
    idx = pd.bdate_range("2009-12-28", "2010-02-26")
    rng = np.random.default_rng(0)
    us = 3 + np.cumsum(rng.normal(scale=0.05, size=len(idx)))
    cn = pd.Series(np.nan, index=idx)
    # 中国 = 前一个美国交易日变化的 0.5 倍
    dus = np.r_[0, np.diff(us)]
    cn.iloc[1:] = 3 + np.cumsum(0.5 * dus[:-1])
    cn.iloc[0] = 3
    return pd.DataFrame({"cn": cn, "us": us}, index=idx)


def test_daily_pairs_recovers_lagged_beta():
    d = daily_pairs(_toy(), "cn", "us")
    assert d.index.min() >= pd.Timestamp("2010-01-01")
    b = np.polyfit(d.d_us, d.d_cn, 1)[0]
    assert abs(b - 0.5) < 1e-9


def test_weekly_pairs_starts_in_2010():
    w = weekly_pairs(_toy(), "cn", "us")
    assert w.index.min() >= pd.Timestamp("2010-01-08") and (w.index.day_of_week == 4).all()


def test_fomc_dates_match_the_official_calendar():
    """决议日只有两种例外：2020-03-15 是周日的紧急降息；其余都是美国工作日。日期核对过美联储官网，2010 年起到 2026-10 共 136 次。"""
    import pandas as pd
    from macrob.rpt_common import DATA
    f = pd.read_csv(DATA / "fomc_dates.csv")
    d = pd.to_datetime(f.us_date)
    assert len(f) == 136 and d.is_unique
    assert list(f.us_date[d.dt.day_of_week >= 5]) == ["2020-03-15"]
    for right in ("2019-03-20", "2020-03-03", "2024-05-01", "2024-12-18"):
        assert right in set(f.us_date)
    for wrong in ("2019-03-21", "2019-05-31", "2020-10-05", "2024-05-02", "2024-12-19", "2026-04-18"):
        assert wrong not in set(f.us_date)
