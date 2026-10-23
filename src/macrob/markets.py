"""日度市场数据：2 年、10 年国债收益率（%），沪深300、中证500 收盘（本机行情库）。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import sources as S

INDEX = S.EXTERNAL / "index_daily" / "all_index.parquet"


@lru_cache(maxsize=1)
def daily() -> pd.DataFrame:
    ix = pd.read_parquet(INDEX, columns=["date", "code", "close"])
    ix = ix[ix["code"].astype(str).isin(["000300", "000905"])]
    px = ix.pivot_table(index=pd.to_datetime(ix["date"]), columns=ix["code"].astype(str), values="close").rename(
        columns={"000300": "hs300", "000905": "zz500"})
    d = px.join(S.yields_daily(), how="left").sort_index()
    d[["y2", "y10"]] = d[["y2", "y10"]].ffill(limit=3)
    return d.dropna(subset=["hs300"])
