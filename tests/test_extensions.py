"""补做部分的关键规则：调息事件的分组、Nelson-Siegel 拟合、实时预测的数据时点、数据意外的窗口和延长、CSMAR 数据的拼接。"""
import numpy as np
import pandas as pd
import pytest

from macrob import credit, curve, nowcast, policy_irs, policy_rates, surprise_nbs, volforecast


def test_rate_events_grouping_follows_the_30_day_rule():
    e = policy_rates.rate_events().set_index("date")
    assert pd.Timestamp("2018-04-17") not in e.index                 # 和降准同一天、方向相反，剔除
    assert e.loc["2019-11-05", "分组"] == "首发"
    assert e.loc["2019-11-18", "分组"] == "跟随"                     # 13 天前 MLF 刚降
    assert e.loc["2024-09-24", "分组"] == "发布会"
    assert e.loc["2023-08-15", "size"] == -10                        # 同一天合并，取 7 天逆回购的幅度


def test_nelson_siegel_recovers_known_factors():
    x = curve.LAM * curve.TAU
    l2 = (1 - np.exp(-x)) / x
    y = 3.0 + (-1.0) * l2 + 0.5 * (l2 - np.exp(-x))                 # 水平 3，斜率（长端减短端）1，曲率 0.5
    f = curve.ns_factors(pd.DataFrame([y], columns=curve.COLS, index=[pd.Timestamp("2020-01-31")]))
    assert np.allclose(f[["水平", "斜率", "曲率"]].iloc[0].to_numpy(), [3.0, 1.0, 0.5])
    assert f["拟合误差(bp)"].iloc[0] < 1e-8


def test_nowcast_vintage_only_contains_published_data():
    idx = pd.date_range("2005-01-31", "2016-12-31", freq="ME")
    raw = pd.DataFrame({k: np.arange(len(idx), dtype=float) for k in nowcast.ACT}, index=idx)
    x = nowcast.vintage(raw, pd.Timestamp("2016-08-31"), pd.Period("2016Q3", "Q"))
    assert x.index.max() == pd.Timestamp("2016-09-30")               # 延伸到季末，留给模型外推
    assert x["pmi"].last_valid_index() == pd.Timestamp("2016-07-31")      # 2016 年 8 月的 PMI 9 月 1 日才发布
    assert x["ip_yoy"].last_valid_index() == pd.Timestamp("2016-07-31")   # 统计局数据次月公布
    idx2 = pd.date_range("2005-01-31", "2018-03-31", freq="ME")
    raw2 = pd.DataFrame({k: np.arange(len(idx2), dtype=float) for k in nowcast.ACT}, index=idx2)
    x2 = nowcast.vintage(raw2, pd.Timestamp("2018-01-31"), pd.Period("2018Q1", "Q"))
    assert x2["pmi"].last_valid_index() == pd.Timestamp("2018-01-31")     # 2018 年 1 月的 PMI 当月 31 日（周三）发布


def test_surprise_window_handles_session_and_weekend_releases():
    days = pd.DatetimeIndex(["2016-04-14", "2016-04-15", "2016-04-18"])        # 周四、周五、下周一
    times = pd.Series(pd.to_datetime(["2016-04-15 10:00", "2016-04-15 16:00", "2016-04-16 10:00"]))
    before, after = surprise_nbs.windows(times, days)
    assert list(before) == [0, 0, 1]
    assert list(after) == [1, 2, 2]          # 盘中发布到当天收盘；收盘后或周末发布顺延到下一个交易日


def test_investing_forecasts_only_change_releases_from_2024():
    old, new = surprise_nbs.releases("固投"), surprise_nbs.releases("固投", extend=True)
    cut = surprise_nbs.EXTEND_FROM
    pd.testing.assert_frame_equal(old[old["时间"] < cut].reset_index(drop=True), new[new["时间"] < cut].reset_index(drop=True))
    late = new[new["时间"] >= cut]
    assert len(late) == 30 and late["forecast"].notna().all()
    assert late.loc[late["时间"] == "2024-03-18 10:00", "forecast"].item() == 3.2      # 1–2 月合并发布，取 Investing 的值


needs_csmar = pytest.mark.skipif(not credit.CSMAR.exists(), reason="CSMAR 的数据有版权，只在本地")


@needs_csmar
def test_irs_cleaning_drops_the_typo_and_keeps_real_jumps():
    r = policy_irs.irs()
    assert pd.Timestamp("2021-08-09") not in r["5Y"].dropna().index                 # 记成 10.43%，前后两天 2.6–2.7%
    assert r.loc["2013-06-20", "1Y"] == 4.82                                         # 钱荒那天是真实行情，保留
    assert abs(r.loc["2021-04-29", "1Y"] - r.loc["2021-04-28", "1Y"]) <= 0.02         # 换字段那天没有跳变


@needs_csmar
def test_tsf_flows_splice_csmar_before_2015():
    f = credit.tsf_flows()
    assert f.index[0] == pd.Timestamp("2012-01-31") and f.loc["2012-01-31"] == 9559
    assert f.loc["2015-01-31"] == 20516                                               # 2015 年起用商务部接口的数


def test_implied_vol_interpolates_variance_to_30_days():
    assert np.isclose(volforecast.interp_30d([(20, 0.2), (50, 0.2)]), 0.2)               # 期限结构平：还是 0.2
    v = volforecast.interp_30d([(20, 0.2), (50, 0.3)])
    assert np.isclose(v, np.sqrt((20 * 0.04 * 20 / 30 + 50 * 0.09 * 10 / 30) / 30))      # 方差按时间线性插值
    assert volforecast.interp_30d([(35, 0.25), (63, 0.3)]) == 0.25                        # 最近的到期月已超过 30 天，直接用


def test_portfolio_weights_drift_within_month_and_reset_next_month():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03"])
    r = pd.DataFrame({"股": [0.10, 0.10, 0.10], "债": [0.0, 0.0, 0.0]}, index=idx)
    p = volforecast.portfolio_daily(r, 0.2)
    assert np.isclose(p.iloc[0], 0.02)
    assert np.isclose(p.iloc[1], 0.22 / 1.02 * 0.10)                                      # 月内股票权重漂到 0.22/1.02
    assert np.isclose(p.iloc[2], 0.02)                                                    # 下个月回到 20%
