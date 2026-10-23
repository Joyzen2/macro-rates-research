"""取数小工具：公开接口常断连，重试几次，取到的原始表存在 data/raw/，之后直接读本地。"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def cached(name: str, fetch, refresh: bool = False) -> pd.DataFrame:
    f = RAW / f"{name}.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f)
    err = None
    for k in range(6):
        try:
            d = fetch()
            d.to_csv(f, index=False)
            return d
        except Exception as e:
            err = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"{name}: {type(err).__name__} {str(err)[:120]}")
