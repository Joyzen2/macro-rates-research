"""第 7 步：信贷脉冲领先经济和市场多久。运行：uv run python -m macrob.credit

规则（看结果之前定的）：
- 信贷脉冲（数据所属月份 m）= （过去 12 个月新增人民币贷款 − 再往前 12 个月的）÷ 一年前的名义 GDP（最近 4 个季度之和）× 100。
  贷款数据次月才公布，所以月末 T 能用的是 m = T − 1 月的值。
- 领先性：月末 T 的信贷脉冲，和 T+k 月的 PMI、工业增加值同比的相关，k = 0–18 个月，找相关最强的 k。
- 对市场：和之后 6 个月、12 个月的沪深300 全收益涨跌、10 年国债收益率变化的秩相关；
  t 值用 Newey-West（滞后阶数 = 持有月数），再用整体平移 12 个月以上的安慰剂（2000 次）算 p 值。
- 稳定性：滚动 36 个月的相关，看哪些年份失效。
第 13 步：同样的检验再用社融增量做一遍（2012–2014 年用本人从学校 CSMAR 下的，2015 年起用商务部接口），结果另存「_社融」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import sources as S
from .allocation import daily_levels
from .datalayer import RESULTS, ROOT, raw_panel

N_PERM = 2000
CSMAR = ROOT / "data" / "manual" / "csmar_20261017"               # 本人从学校 CSMAR 下的，有版权，只在本地用


def ngdp_4q() -> pd.Series:
    d = S.em("macro_china_gdp")
    q = d["季度"].str.extract(r"(\d{4})年第1-?(\d)?季度")
    year = q[0].astype(int)
    last_q = q[1].fillna("1").astype(int)
    cum = pd.Series(d["国内生产总值-绝对值"].astype(float).to_numpy(), index=pd.MultiIndex.from_arrays([year, last_q]))
    cum = cum[~cum.index.duplicated()].sort_index()
    quarterly = {}
    for (y, k), v in cum.items():
        prev = cum.get((y, k - 1), 0.0) if k > 1 else 0.0
        quarterly[pd.Timestamp(y, 3 * k, 1) + pd.offsets.MonthEnd(0)] = v - prev
    s = pd.Series(quarterly).sort_index()
    return s.rolling(4).sum().dropna()


def tsf_flows() -> pd.Series:
    """社融当月增量（亿元）：2012–2014 年用 CSMAR 的，2015 年起用商务部接口。"""
    old = pd.read_csv(CSMAR / "社融当月增量_201201_201412_已核对.csv")
    old = pd.Series(old["tsf_flow_100m_cny"].astype(float).to_numpy(), index=pd.to_datetime(old["month"]) + pd.offsets.MonthEnd(0))
    new = S.em("macro_china_shrzgm")
    new = pd.Series(new["社会融资规模增量"].astype(float).to_numpy(),
                    index=pd.to_datetime(new["月份"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0))
    flow = pd.concat([old[old.index < "2015-01-01"], new[new.index >= "2015-01-01"]]).sort_index()
    assert flow.index.equals(pd.date_range(flow.index[0], flow.index[-1], freq="ME")), "社融月份不连续"
    return flow


def _impulse(flow: pd.Series) -> pd.DataFrame:
    f12 = flow.rolling(12).sum()
    g = ngdp_4q().resample("ME").last().ffill()
    ci = (f12 - f12.shift(12)) / g.shift(12).reindex(f12.index).ffill() * 100
    out = pd.DataFrame({"数据月份的信贷脉冲": ci})
    out["月末可用的信贷脉冲"] = out["数据月份的信贷脉冲"].shift(1)        # 次月公布
    return out.dropna()


def impulse() -> pd.DataFrame:
    return _impulse(S._series(S.em("macro_china_new_financial_credit"), "月份", "当月"))


def tsf_impulse() -> pd.DataFrame:
    return _impulse(tsf_flows())


def _nw_t(x: pd.Series, y: pd.Series, lags: int) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    r = sm.OLS(d.iloc[:, 1].rank(), sm.add_constant(d.iloc[:, 0].rank())).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.tvalues.iloc[1])


def _ic(x, y):
    d = pd.concat([x, y], axis=1).dropna()
    return float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))


def study(ci: pd.Series, suffix: str) -> None:
    raw = raw_panel()
    lead_rows = []
    for name, tgt in (("PMI", raw["pmi"]), ("工业增加值同比", raw["ip_yoy"].resample("ME").last().interpolate(limit=1))):
        for k in range(0, 19):
            y = tgt.shift(-k).reindex(ci.index)
            lead_rows.append({"目标": name, "领先月数": k, "相关": _ic(ci, y)})
    leads = pd.DataFrame(lead_rows)
    leads.to_csv(RESULTS / f"credit_impulse_leads{suffix}.csv", index=False, encoding="utf-8-sig")
    lv = daily_levels().resample("ME").last()
    y10 = S.yields_daily()["y10"].resample("ME").last()
    rng = np.random.default_rng(0)
    rows = []
    for h in (6, 12):
        tg = {f"之后{h}个月沪深300(%)": (lv["股"].shift(-h) / lv["股"] - 1) * 100, f"之后{h}个月Δ10年(bp)": (y10.shift(-h) - y10) * 100}
        for tname, y in tg.items():
            y = y.reindex(ci.index)
            ic = _ic(ci, y)
            n = len(ci)
            fake = np.array([_ic(pd.Series(np.roll(ci.to_numpy(), k), index=ci.index), y) for k in rng.integers(12, n - 12, size=N_PERM)])
            rows.append({"目标": tname, "月数": int(pd.concat([ci, y], axis=1).dropna().shape[0]), "秩相关": ic,
                         "Newey-West t": _nw_t(ci, y, h), "平移安慰剂 p": float((np.abs(fake) >= abs(ic)).mean())})
    mk = pd.DataFrame(rows)
    mk.to_csv(RESULTS / f"credit_impulse_markets{suffix}.csv", index=False, encoding="utf-8-sig")
    best = leads.loc[leads.groupby("目标")["相关"].idxmax()]
    pmi_k = int(best.loc[best["目标"] == "PMI", "领先月数"].iloc[0])
    roll = ci.rolling(36).corr(raw["pmi"].shift(-pmi_k).reindex(ci.index))
    weak = roll[roll < 0]
    pd.set_option("display.width", 220)
    print(f"===== {'社融' if suffix else '新增贷款'}口径：信贷脉冲 {ci.index.min():%Y-%m} ~ {ci.index.max():%Y-%m}，最新 {ci.iloc[-1]:.2f}")
    print("领先性（每个目标相关最强的领先月数）：\n", best.round(3).to_string(index=False))
    print("PMI 不同领先月数的相关：", leads[leads["目标"] == "PMI"].set_index("领先月数")["相关"].round(2).to_dict())
    print(mk.round(3).to_string(index=False))
    print(f"滚动 36 个月相关（领先 {pmi_k} 个月）为负的时段：", f"{weak.index.min():%Y-%m} ~ {weak.index.max():%Y-%m}，共 {len(weak)} 个月" if len(weak) else "没有")


def main() -> None:
    study(impulse()["月末可用的信贷脉冲"], "")
    study(tsf_impulse()["月末可用的信贷脉冲"], "_社融")


if __name__ == "__main__":
    main()
