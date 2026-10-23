"""第 12 步（补数据）：BIS 总信贷的季度信贷脉冲。运行：uv run python -m macrob.credit_bis

规则（看结果之前定的，写在计划文件第 12 步里）：
- 数据：BIS 统计的中国私人非金融部门总信贷（调整断点后的季末存量，十亿元，含贷款和债券），从 FRED 下载（CRDQCNAPABIS）。
- 信贷脉冲（季度 q）=（过去 4 个季度新增信贷 − 再往前 4 个季度的）÷ 一年前的名义 GDP（4 个季度之和）× 100。
- BIS 大约晚两个季度公布：季度末 T 只用 T − 2 季度的值。
- 领先性：和 0–6 个季度后 PMI、工业增加值同比（季度平均）的相关。
- 对市场：之后 2 个、4 个季度沪深300 全收益、10 年收益率变化的秩相关；Newey-West t（滞后 = 持有季度数）；平移安慰剂 2000 次。
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from . import credit, sources as S
from .allocation import daily_levels
from .datalayer import RESULTS, raw_panel
from .probe_util import cached

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CRDQCNAPABIS"
N_PERM, LAG_Q = 2000, 2


def bis_credit() -> pd.Series:
    d = cached("fred_CRDQCNAPABIS", lambda: pd.read_csv(io.StringIO(requests.get(FRED, timeout=30).text)))
    s = pd.Series(d.iloc[:, 1].astype(float).to_numpy(), index=pd.to_datetime(d.iloc[:, 0]) + pd.offsets.QuarterEnd(0))
    return s.sort_index()


def impulse() -> pd.DataFrame:
    c = bis_credit() * 10                                       # 十亿元 → 亿元，和 GDP 同单位
    flow = c - c.shift(4)                                       # 过去 4 个季度的新增信贷
    g = credit.ngdp_4q()
    g.index = g.index + pd.offsets.QuarterEnd(0)
    ci = (flow - flow.shift(4)) / g.shift(4).reindex(flow.index) * 100
    out = pd.DataFrame({"数据季度的信贷脉冲": ci, "信贷占 GDP": c / g.reindex(c.index) * 100})
    out["季末可用的信贷脉冲"] = out["数据季度的信贷脉冲"].shift(LAG_Q)
    return out


def nw_rank_t(x: pd.Series, y: pd.Series, lags: int) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    r = sm.OLS(d.iloc[:, 1].rank(), sm.add_constant(d.iloc[:, 0].rank())).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.tvalues.iloc[1])


def main() -> None:
    ci_df = impulse()
    ci = ci_df["季末可用的信贷脉冲"].dropna()
    raw = raw_panel()
    q_pmi = raw["pmi"].resample("QE").mean()
    q_ip = raw["ip_yoy"].resample("QE").mean()
    leads = []
    for name, tgt in (("PMI", q_pmi), ("工业增加值同比", q_ip)):
        for k in range(0, 7):
            d = pd.concat([ci, tgt.shift(-k)], axis=1).dropna()
            leads.append({"目标": name, "领先季度数": k, "相关": float(d.iloc[:, 0].corr(d.iloc[:, 1])), "季度数": len(d)})
    leads = pd.DataFrame(leads)
    stock = daily_levels()["股"].resample("QE").last()
    y10 = S.yields_daily()["y10"].resample("QE").last()
    rng = np.random.default_rng(0)
    rows = []
    for h in (2, 4):
        for name, fwd in ((f"之后 {h} 个季度沪深300(%)", (stock.shift(-h) / stock - 1) * 100), (f"之后 {h} 个季度 Δ10年(bp)", (y10.shift(-h) - y10) * 100)):
            d = pd.concat([ci, fwd], axis=1).dropna()
            ic = float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
            x, y = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
            fake = np.array([pd.Series(np.roll(x, k)).corr(pd.Series(y), method="spearman") for k in rng.integers(4, len(x) - 4, size=N_PERM)])
            rows.append({"目标": name, "季度数": len(d), "起": f"{d.index.min():%Y-%m}", "秩相关": ic, "Newey-West t": nw_rank_t(d.iloc[:, 0], d.iloc[:, 1], h),
                         "平移安慰剂 p": float((np.abs(fake) >= abs(ic)).mean())})
    mk = pd.DataFrame(rows)
    ci_df.to_csv(RESULTS / "credit_impulse_bis.csv", encoding="utf-8-sig")
    leads.to_csv(RESULTS / "credit_impulse_bis_leads.csv", index=False, encoding="utf-8-sig")
    mk.to_csv(RESULTS / "credit_impulse_bis_markets.csv", index=False, encoding="utf-8-sig")
    # 和贷款口径的月度信贷脉冲对比（季末取值）
    loans = credit.impulse()["数据月份的信贷脉冲"].resample("QE").last()
    both = pd.concat([ci_df["数据季度的信贷脉冲"], loans], axis=1, keys=["BIS 总信贷", "新增贷款"]).dropna()
    pd.set_option("display.width", 200)
    print(f"BIS 信贷脉冲 {ci.index.min():%Y-%m} ~ {ci.index.max():%Y-%m}，最新（数据季度）{ci_df['数据季度的信贷脉冲'].dropna().iloc[-1]:.2f}；"
          f"和贷款口径的相关 {both.corr().iloc[0, 1]:.2f}（{len(both)} 个季度）")
    print(leads.pivot(index="领先季度数", columns="目标", values="相关").round(3).to_string())
    print(mk.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
