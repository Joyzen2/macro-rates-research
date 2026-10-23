"""数据层：指标登记表 → 月度序列 → 实时面板。运行：uv run python -m macrob.datalayer

实时面板的规则（看结果之前定的）：每个月末，每个指标只取「已经公布」的最新一期。
  滞后 0：当月月末就能用（PMI 在当月最后一天发布；市场数据取当月月末值或月均）。
  滞后 1：次月月末才能用（统计局、海关、央行的月度数据一般在次月 7–17 日发布）。
  滞后 2：再晚一个月（消费者信心指数发布较晚，保守处理）。
拿到的是现在网站上的数，不是当年的首次公布值；同比数据统计局很少修订，这里用金十的首次公布值做了核对（见 revision_check）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import sources as S
from .probe_util import RAW

ROOT = RAW.parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


@dataclass
class Indicator:
    key: str
    name: str
    category: str
    source: str
    lag: int
    get: Callable[[], pd.Series]


def _pmi(row):
    return lambda: S.nbs_row("pmi", S.PMI_PATH, row)


def _em(fn, date_col, val_col):
    return lambda: S._series(S.em(fn), date_col, val_col)


def _nbs(name, path, row):
    return lambda: S.nbs_row(name, path, row)


def _monthly_mean(daily: pd.Series) -> pd.Series:
    return daily.resample("ME").mean()


def _pmi_no_inv():
    return _pmi(r"^新订单")() - _pmi(r"^产成品库存")()


def _trade_balance():
    d = S.em("macro_china_hgjck")
    x, m = S._series(d, "月份", "当月出口额-金额"), S._series(d, "月份", "当月进口额-金额")
    return (x - m) / (x + m) * 100


def _loans_12m():
    s = S._series(S.em("macro_china_new_financial_credit"), "月份", "当月")
    roll = s.rolling(12).sum()
    return (roll / roll.shift(12) - 1) * 100


def _money(col):
    return _em("macro_china_money_supply", "月份", col)


def _m1_m2():
    return _money("货币(M1)-同比增长")() - _money("货币和准货币(M2)-同比增长")()


def _y10():
    return S.yields_daily()["y10"].resample("ME").last()


def _term():
    y = S.yields_daily().resample("ME").last()
    return (y["y10"] - y["y2"]) * 100


REGISTRY = [
    # 生产
    Indicator("ip_yoy", "工业增加值当月同比", "生产", "东方财富", 1, _em("macro_china_gyzjz", "月份", "同比增长")),
    Indicator("pmi", "制造业 PMI", "生产", "国家统计局", 0, _pmi(r"^制造业采购经理指数")),
    Indicator("pmi_prod", "PMI 生产", "生产", "国家统计局", 0, _pmi(r"^生产指数")),
    Indicator("pmi_new", "PMI 新订单", "生产", "国家统计局", 0, _pmi(r"^新订单")),
    Indicator("pmi_no_inv", "PMI 新订单减产成品库存", "生产", "国家统计局（推算）", 0, _pmi_no_inv),
    Indicator("power_yoy", "发电量当月同比", "生产", "国家统计局", 1, _nbs("power", "能源 > 能源主要产品产量 > 发电量", r"发电量_同比增长")),
    # 投资
    Indicator("fai_ytd", "固定资产投资累计同比", "投资", "国家统计局", 1, _nbs("fai_overview", "固定资产投资 (不含农户)  > 固定资产投资概况", r"^固定资产投资额累计增长")),
    Indicator("re_inv_ytd", "房地产开发投资累计同比", "投资", "国家统计局", 1, _nbs("re_invest", "房地产 > 房地产开发投资情况", r"^房地产投资_累计增长")),
    Indicator("re_start_ytd", "新开工面积累计同比", "投资", "国家统计局", 1, _nbs("re_area", "房地产 > 房地产施工、竣工面积", r"新开工施工面积累计增长")),
    Indicator("re_sales_ytd", "商品房销售面积累计同比", "投资", "国家统计局", 1, _nbs("re_sales", "房地产 > 新建商品房销售面积", r"累计增长")),
    # 消费
    Indicator("retail_yoy", "社会消费品零售总额当月同比", "消费", "东方财富", 1, _em("macro_china_consumer_goods_retail", "月份", "同比增长")),
    Indicator("confidence", "消费者信心指数", "消费", "东方财富", 2, _em("macro_china_xfzxx", "月份", "消费者信心指数-指数值")),
    Indicator("autos_yoy", "汽车产量当月同比", "消费", "国家统计局", 1, _nbs("autos", "工业 > 工业主要产品产量 > 汽车", r"汽车产量_同比增长")),
    Indicator("services_yoy", "服务业生产指数当月同比", "消费", "国家统计局", 1, _nbs("services", "服务业生产指数 > 服务业生产指数", r"当月同比")),
    # 进出口
    Indicator("export_yoy", "出口当月同比（美元）", "进出口", "东方财富", 1, _em("macro_china_hgjck", "月份", "当月出口额-同比增长")),
    Indicator("import_yoy", "进口当月同比（美元）", "进出口", "东方财富", 1, _em("macro_china_hgjck", "月份", "当月进口额-同比增长")),
    Indicator("trade_bal", "贸易差额占进出口总额", "进出口", "东方财富（推算）", 1, _trade_balance),
    Indicator("pmi_export", "PMI 新出口订单", "进出口", "国家统计局", 0, _pmi(r"^新出口订单")),
    # 信用
    Indicator("loans_12m", "新增人民币贷款（12 个月滚动和）同比", "信用", "东方财富（推算）", 1, _loans_12m),
    Indicator("m1_yoy", "M1 同比", "信用", "东方财富", 1, _money("货币(M1)-同比增长")),
    Indicator("m2_yoy", "M2 同比", "信用", "东方财富", 1, _money("货币和准货币(M2)-同比增长")),
    Indicator("m1_m2", "M1 与 M2 增速之差", "信用", "东方财富（推算）", 1, _m1_m2),
    # 货币
    Indicator("fr007", "7 天回购定盘利率（月均）", "货币", "中国外汇交易中心", 0, lambda: _monthly_mean(S.repo7d_daily())),
    Indicator("shibor3m", "3 个月 SHIBOR（月均）", "货币", "上海银行间同业拆放利率", 0, lambda: _monthly_mean(S.shibor3m_daily())),
    Indicator("y10", "10 年国债收益率（月末）", "货币", "东方财富", 0, _y10),
    Indicator("term", "10 年减 2 年期限利差（基点，月末）", "货币", "东方财富（推算）", 0, _term),
    # 通胀
    Indicator("cpi_yoy", "CPI 同比", "通胀", "东方财富", 1, _em("macro_china_cpi", "月份", "全国-同比增长")),
    Indicator("ppi_yoy", "PPI 同比", "通胀", "东方财富", 1, _em("macro_china_ppi", "月份", "当月同比增长")),
    Indicator("pmi_outp", "PMI 出厂价格", "通胀", "国家统计局", 0, _pmi(r"^出厂价格")),
    Indicator("pmi_inp", "PMI 原材料购进价格", "通胀", "国家统计局", 0, _pmi(r"^主要原材料购进价格")),
]
LAGS = {i.key: i.lag for i in REGISTRY}
CATEGORIES = ["生产", "投资", "消费", "进出口", "信用", "货币", "通胀"]


def raw_panel() -> pd.DataFrame:
    """数据所属月份 × 指标（还没按发布滞后处理）。"""
    f = ROOT / "data" / "raw_panel.parquet"
    if f.exists():
        return pd.read_parquet(f)
    cols = {}
    for ind in REGISTRY:
        s = ind.get()
        s.index = s.index + pd.offsets.MonthEnd(0)
        cols[ind.key] = s[~s.index.duplicated(keep="last")]
    p = pd.DataFrame(cols).sort_index()
    p = p[p.index >= "2005-01-31"]
    p.to_parquet(f)
    return p


PMI_KEYS = ("pmi", "pmi_prod", "pmi_new", "pmi_no_inv", "pmi_export", "pmi_outp", "pmi_inp")


def _last_trading_day() -> pd.Series:
    """每个月最后一个交易日：中债曲线有数的日子去掉周六日（调休的周六银行间开市，交易所不开）。"""
    from .probe_util import RAW
    days = pd.concat([pd.read_csv(RAW / f"chinabond_curve_{y}.csv", usecols=["日期"]) for y in range(2006, 2027)])["日期"]
    days = pd.DatetimeIndex(pd.to_datetime(days).unique()).sort_values()
    days = days[days.dayofweek < 5]
    s = pd.Series(days, index=days)
    return s.groupby(s.index + pd.offsets.MonthEnd(0)).max()


def pmi_lag() -> pd.Series:
    """每个数据月份的 PMI 在月末能不能用：0 = 当月月末能用，1 = 要到次月月末。

    修正（看完结果后发现）：原来一律按 0。实际上 2005–2016 年统计局在次月 1 日发布；2017 年起在当月最后一天，
    但最后一天常是周末，那时当月最后一个交易日已经收盘。规则：发布日不晚于当月最后一个交易日才算 0。
    发布日来自金十；金十没有的月份按当月最后一天发布算；2006-03 以前没有交易日历，按发布日是否晚于月末判断。
    """
    d = pd.read_csv(S.UPGRADE_DATA / "consensus_macro_china_pmi_yearly.csv", parse_dates=["日期"]).dropna(subset=["今值"])
    rel = pd.Series(d["日期"].to_numpy(), index=(d["日期"] - pd.Timedelta(days=5)) + pd.offsets.MonthEnd(0))
    rel = rel[~rel.index.duplicated(keep="first")]
    months = pd.date_range("2005-01-31", pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0), freq="ME")
    rel = rel.reindex(months).fillna(pd.Series(months, index=months))
    ltd = _last_trading_day().reindex(months).fillna(pd.Series(months, index=months))
    return (rel > ltd).astype(int)


def available_at(key: str, months: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """数据月份 → 第一次能用的月末。"""
    if key in PMI_KEYS:
        lag = pmi_lag()
        return pd.DatetimeIndex([m + pd.offsets.MonthEnd(int(lag.get(m, 0))) for m in months])
    return months + pd.offsets.MonthEnd(LAGS[key])


CSMAR_MACRO = ROOT / "data" / "manual" / "csmar_monthly_macro_20261019" / "csmar_monthly_macro_clean_2000-01_to_2026-08.csv"
CSMAR_COLS = {"export_yoy": "export_yoy_pct", "import_yoy": "import_yoy_pct", "trade_bal": "trade_balance_share_pct",
              "cpi_yoy": "cpi_yoy_pct", "ppi_yoy": "ppi_yoy_pct"}


def with_csmar_history(raw: pd.DataFrame) -> pd.DataFrame:
    """第 14 步：这五个指标在现有序列开始之前的月份（2005 年起），用本人从学校 CSMAR 下的数补上，重合的月份不动。
    只给状态识别用（PCA 的预热期）；GDP 实时预测、泰勒规则用原来的数据。CSMAR 的文件不在时原样返回。"""
    if not CSMAR_MACRO.exists():
        return raw
    c = pd.read_csv(CSMAR_MACRO)
    c.index = pd.to_datetime(c["month"]) + pd.offsets.MonthEnd(0)
    fills = {}
    for k, col in CSMAR_COLS.items():
        h = c[col].astype(float).dropna()
        fills[k] = h[(h.index >= raw.index.min()) & (h.index < raw[k].first_valid_index())]
    raw = raw.reindex(raw.index.union(pd.DatetimeIndex(sorted({t for f in fills.values() for t in f.index}))))
    for k, f in fills.items():
        raw.loc[f.index, k] = f.to_numpy()
    return raw


def realtime_panel(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """月末 T × 指标：T 当时已经公布的最新一期（按滞后规则），没有就向前沿用上一期。"""
    raw = raw_panel() if raw is None else raw
    out = {}
    last = min(raw.index.max() + pd.offsets.MonthEnd(2), pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1))   # 只到上一个已经过完的月末
    idx = pd.date_range(raw.index.min(), last, freq="ME")
    for k in raw.columns:
        s = raw[k].dropna()
        s = pd.Series(s.to_numpy(), index=available_at(k, s.index)).groupby(level=0).last()   # 同一个月末能用两期时取新的
        out[k] = s.reindex(idx).ffill(limit=3)
    return pd.DataFrame(out, index=idx)


def dictionary() -> pd.DataFrame:
    raw = raw_panel()
    rows = []
    for i in REGISTRY:
        s = raw[i.key].dropna()
        rows.append({"类别": i.category, "代码": i.key, "指标": i.name, "来源": i.source,
                     "发布滞后（月）": "0 或 1，看发布日" if i.key in PMI_KEYS else i.lag,
                     "起": s.index.min().strftime("%Y-%m"), "止": s.index.max().strftime("%Y-%m"), "月数": len(s),
                     "缺失月数（起止之间）": int(len(pd.date_range(s.index.min(), s.index.max(), freq="ME")) - len(s))})
    return pd.DataFrame(rows)


def quality() -> pd.DataFrame:
    """异常跳变：月度变化超过该指标月度变化标准差 6 倍的月份；1 月缺失（1–2 月合并发布）的次数。"""
    raw = raw_panel()
    rows = []
    for k in raw.columns:
        s = raw[k].dropna()
        d = s.diff()
        big = d[(d - d.median()).abs() > 6 * d.std()]
        jan_missing = sum(1 for y in range(s.index.min().year + 1, s.index.max().year + 1)
                          if pd.Timestamp(y, 1, 31) not in s.index and pd.Timestamp(y, 2, 28) + pd.offsets.MonthEnd(0) in s.index)
        rows.append({"代码": k, "异常跳变月份": "；".join(x.strftime("%Y-%m") for x in big.index[:6]), "1 月缺失年数": jan_missing})
    return pd.DataFrame(rows)


def revision_check() -> pd.DataFrame:
    """首次公布值（金十，第一轮缓存）和现在网站上的数对比。"""
    pairs = {"ip_yoy": "macro_china_industrial_production_yoy", "cpi_yoy": "macro_china_cpi_yearly", "ppi_yoy": "macro_china_ppi_yearly",
             "pmi": "macro_china_pmi_yearly", "export_yoy": "macro_china_exports_yoy", "import_yoy": "macro_china_imports_yoy"}
    raw = raw_panel()
    rows = []
    for k, fn in pairs.items():
        f = S.UPGRADE_DATA / f"consensus_{fn}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["month"] = (pd.to_datetime(d["日期"]) - pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)
        if k == "pmi":                                        # PMI 在当月最后一天或次月 1 日发布：往前推 5 天再取月份
            d["month"] = (pd.to_datetime(d["日期"]) - pd.Timedelta(days=5)) + pd.offsets.MonthEnd(0)
        first = d.dropna(subset=["今值"]).drop_duplicates("month", keep="last").set_index("month")["今值"].astype(float)
        both = pd.concat([first.rename("首次"), raw[k].rename("现在")], axis=1).dropna()
        both = both[both.index >= "2010-01-31"]
        diff = (both["现在"] - both["首次"]).abs()
        rows.append({"代码": k, "对比月数": len(both), "平均绝对差": diff.mean(), "差超过 0.1 的比例": float((diff > 0.1).mean()),
                     "最大差": diff.max(), "最大差的月份": diff.idxmax().strftime("%Y-%m") if len(diff) else ""})
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 240)
    d = dictionary(); d.to_csv(RESULTS / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    print(d.to_string(index=False))
    q = quality(); q.to_csv(RESULTS / "data_quality.csv", index=False, encoding="utf-8-sig")
    print(q[q["异常跳变月份"].ne("") | q["1 月缺失年数"].gt(0)].to_string(index=False))
    r = revision_check(); r.to_csv(RESULTS / "revision_check.csv", index=False, encoding="utf-8-sig")
    print(r.round(3).to_string(index=False))
    rt = realtime_panel(); rt.to_parquet(ROOT / "data" / "realtime_panel.parquet")
    print("实时面板", rt.shape, rt.index.min().date(), "~", rt.index.max().date())


if __name__ == "__main__":
    main()
