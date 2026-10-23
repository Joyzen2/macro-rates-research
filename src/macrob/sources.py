"""公开数据源。每个函数返回一条月度序列：index = 数据所属月份的月末，值 = 公布的数。

来源：国家统计局数据库（经 akshare）、东方财富（经 akshare）、本机已有的 SHIBOR 和国债收益率。
原始表都缓存在 data/raw/，重跑不会重复下载。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .probe_util import RAW, cached

EXTERNAL = Path(os.environ.get("MACROB_EXTERNAL", RAW.parent / "external"))   # 不在仓库里的外部数据，放法见 README
UPGRADE_DATA = EXTERNAL / "round1" / "data"      # 第一轮缓存的国债收益率、金十数据
CLOCKOPT_DATA = EXTERNAL / "shibor"              # 3 个月 SHIBOR


def _month_end(s: pd.Series) -> pd.DatetimeIndex:
    """'2026年08月份' / '2026年8月' / '202608' / '2026-08-01' / '2026.8' → 月末。"""
    def one(x):
        x = str(x).strip()
        m = re.match(r"(\d{4})\D*?(\d{1,2})", x) if not re.fullmatch(r"\d{6}", x) else re.match(r"(\d{4})(\d{2})", x)
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex([one(x) for x in s])


def _series(df: pd.DataFrame, date_col: str, val_col: str) -> pd.Series:
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").to_numpy(), index=_month_end(df[date_col]))
    return s[~s.index.duplicated(keep="first")].sort_index().dropna()


# ── 国家统计局 ─────────────────────────────────────────────────────

def _nbs_fetch(path: str) -> pd.DataFrame:
    import akshare as ak
    d = ak.macro_china_nbs_nation(kind="月度数据", path=path, period="2005-")
    return d.reset_index().rename(columns={"index": "指标"})


@lru_cache(maxsize=64)
def nbs_table(name: str, path: str) -> pd.DataFrame:
    """行 = 指标，列 = 月份。"""
    d = cached(f"nbs_{name}", lambda: _nbs_fetch(path))
    return d.set_index(d.columns[0])


def nbs_row(name: str, path: str, row_pattern: str) -> pd.Series:
    t = nbs_table(name, path)
    hits = [r for r in t.index if re.search(row_pattern, str(r))]
    if not hits:
        raise KeyError(f"{name}: 没有匹配 {row_pattern} 的行，现有 {list(t.index)[:10]}")
    row = t.loc[hits[0]]
    s = pd.Series(pd.to_numeric(row.to_numpy(), errors="coerce"), index=_month_end(pd.Series(t.columns)))
    return s.sort_index().dropna()


PMI_PATH = "采购经理指数 > 制造业采购经理指数"
CPI_PATHS = {"2026-": "全国居民消费价格分类指数 (上年同月=100) (2026-)", "2021-2025": "全国居民消费价格分类指数 (上年同月=100) (2021-2025)",
             "2016-2020": "全国居民消费价格分类指数 (上年同月=100) (2016-2020) ", "-2015": "全国居民消费价格分类指数 (上年同月=100) (-2015) "}


def core_cpi_yoy() -> pd.Series:
    """不包括食品和能源的 CPI 同比（%）。统计局按年份段分表，这里拼起来。指数是上年同月 = 100，减 100 得同比。"""
    parts = []
    for k, sub in CPI_PATHS.items():
        try:
            parts.append(nbs_row(f"cpi_{k}", f"价格指数 > 居民消费价格分类指数 (上年同月=100)  > {sub}", r"不包括食品和能源"))
        except Exception:
            continue
    s = pd.concat(parts).sort_index()
    return (s[~s.index.duplicated(keep="first")] - 100).rename("核心CPI同比")


# ── 东方财富 ───────────────────────────────────────────────────────

def em(fn: str) -> pd.DataFrame:
    import akshare as ak
    return cached(fn, getattr(ak, fn))


# ── 市场数据（月末值）────────────────────────────────────────────────

def yields_daily() -> pd.DataFrame:
    y = pd.read_csv(UPGRADE_DATA / "bond_zh_us_rate.csv", parse_dates=["日期"]).set_index("日期")
    return y[["中国国债收益率2年", "中国国债收益率10年"]].rename(columns={"中国国债收益率2年": "y2", "中国国债收益率10年": "y10"})


def shibor3m_daily() -> pd.Series:
    return pd.read_csv(CLOCKOPT_DATA / "shibor_3m.csv", parse_dates=["date"]).set_index("date")["rate"]


def repo7d_daily() -> pd.Series:
    """7 天回购定盘利率 FR007（中国外汇交易中心）。"""
    import akshare as ak
    def fetch():
        out = []
        for y in range(2006, pd.Timestamp.today().year + 1):
            try:
                d = ak.repo_rate_hist(start_date=f"{y}0101", end_date=f"{y}1231")
                out.append(d)
            except Exception:
                continue
        return pd.concat(out, ignore_index=True)
    d = cached("repo_rate_hist", fetch)
    col = next(c for c in d.columns if "FR007" in str(c))
    dc = next(c for c in d.columns if "date" in str(c).lower() or "日期" in str(c))
    return pd.Series(pd.to_numeric(d[col], errors="coerce").to_numpy(), index=pd.to_datetime(d[dc])).sort_index().dropna()
