"""仓库里没有数据：需要本地数据的测试，在 data/raw/ 里没有数据文件的时候跳过（导入代码时会自动建一个空的 data/raw/）。"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEEDS_DATA = {
    "test_realtime_panel_respects_publication_lag",
    "test_pmi_lag_uses_actual_release_dates",
    "test_nowcast_vintage_only_contains_published_data",
    "test_investing_forecasts_only_change_releases_from_2024",
}


def pytest_collection_modifyitems(config, items):
    if any((ROOT / "data" / "raw").glob("*.csv")):
        return
    for item in items:
        if item.name in NEEDS_DATA:
            item.add_marker(pytest.mark.skip(reason="需要本地数据，仓库里没有"))
