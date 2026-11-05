import numpy as np
import pandas as pd

from macrob.rpt_futures_basis import annualized_basis, dividend_rate, seasonal_summary, trading_days_to


def test_annualized_basis_sign_and_scale():
    # 期货比现货低 1%，剩 36.5 天 → 年化 −10%
    assert np.isclose(annualized_basis(99.0, 100.0, 36.5), -10.0)
    assert annualized_basis(101.0, 100.0, 365) > 0


def test_dividend_rate():
    # 价格指数不动，全收益涨 2% → 区间分红率 2%
    assert np.isclose(dividend_rate(100, 102, 50, 50), 0.02)
    # 两者同涨 → 没有分红
    assert np.isclose(dividend_rate(100, 110, 50, 55), 0.0)


def test_exdiv_identity():
    basis, div_ann = -6.0, 4.0
    exdiv = basis + div_ann
    assert np.isclose(basis, -div_ann + exdiv)          # 年化基差 = 分红部分 + 去分红部分


def test_trading_days_to():
    cal = pd.bdate_range("2026-01-05", "2026-01-30")
    assert trading_days_to(cal, pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-16")) == 4


def test_seasonal_summary_gap():
    rows = [{"product": "IF", "month": m, "basis_mean_same": -10.0 if 5 <= m <= 8 else -2.0,
             "exdiv_mean": -1.0} for m in range(1, 13)]
    s = seasonal_summary(pd.DataFrame(rows)).iloc[0]
    assert np.isclose(s.basis_gap, -8.0) and np.isclose(s.exdiv_gap, 0.0)


def test_carry_does_not_roll_back_into_an_expiring_contract():
    """快到期的合约换走以后，就算它当天还是持仓最大的，也不能再换回去。"""
    from macrob.rpt_common import expiry_table, load, load_futures
    import macrob.rpt_futures_basis as fb
    fut = load_futures()
    daily = fb.build_daily(fut, load("index_price"), load("index_total_return"))
    navs, rolls = fb.carry(fut, daily, load("index_total_return"), load("funding_daily")["SHIBOR_3M"].dropna())
    exp = expiry_table(fut).set_index("contract")["expiry"]
    for prod, g in daily.groupby("product"):
        cal = pd.DatetimeIndex(g.date)
        r = rolls[rolls["product"] == prod]
        for x in r.itertuples():
            assert fb.trading_days_to(cal, pd.Timestamp(x.date), exp[x.to]) > fb.ROLL_TDAYS, x
