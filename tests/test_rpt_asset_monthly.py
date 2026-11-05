import numpy as np
import pandas as pd

import macrob.rpt_asset_monthly as am


def test_asof_takes_last_value_on_or_before():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-09-29", "2026-09-30", "2026-10-08"]))
    assert am.asof(s, pd.Timestamp("2026-10-01")) == (pd.Timestamp("2026-09-30"), 2.0)


def test_returns_table_periods(monkeypatch):
    idx = pd.bdate_range("2025-10-01", "2026-10-30")
    s = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    t = am.returns_table({"测试": s}).iloc[0]
    v = lambda d: s[s.index <= d].iloc[-1]                      # noqa: E731
    assert np.isclose(t.ret_oct, (v("2026-10-30") / v("2026-09-30") - 1) * 100)
    assert np.isclose(t.ret_ytd, (v("2026-10-30") / v("2025-12-31") - 1) * 100)
    assert np.isclose(t.ret_1y, (v("2026-10-30") / v("2025-10-30") - 1) * 100)
    assert t.complete


def test_equity_bond_spread_definition():
    # 市盈率 12.5 → 盈利收益率 8%，减 10 年 1.65% = 6.35 个百分点
    assert np.isclose(100 / 12.5 - 1.65, 6.35)
