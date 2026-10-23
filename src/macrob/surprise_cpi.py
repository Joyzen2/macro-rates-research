"""第 12 步（补数据续）：CPI 的预期 2025-09 起用 Investing（本人找来的），接在金十后面。运行：uv run python -m macrob.surprise_cpi

规则（看结果之前定的，写在计划文件第 12 步「补数据（续）」里）：
- 金十部分的处理和第一轮一样：去掉公布值或预测值缺失的，同一天留最后一条；金十最后一条之后接 Investing。
- 意外 = 公布值 − 预期值，除以之前所有发布的意外标准差（至少 12 次）。
- 统计局 9:30 发布：窗口 = 发布日前一个交易日收盘 → 发布日收盘；周末发布顺延到下一个交易日。
- 报全样本、截尾 ±3、去掉 2020–2021 三种；原来到 2025-08 的样本用同一套代码重算，并和第一轮的事件表核对。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import markets, sources as S
from .datalayer import RESULTS
from .surprise_nbs import MANUAL, MIN_HIST, reactions, single

ROUND1_EVENTS = S.UPGRADE_DATA.parent / "results" / "surprise_events.csv"


def releases() -> pd.DataFrame:
    j = pd.read_csv(S.UPGRADE_DATA / "consensus_macro_china_cpi_yearly.csv")
    j = j.assign(date=pd.to_datetime(j["日期"]), actual=pd.to_numeric(j["今值"], errors="coerce"),
                 forecast=pd.to_numeric(j["预测值"], errors="coerce"))
    j = j.dropna(subset=["actual", "forecast"]).drop_duplicates("date", keep="last").sort_values("date")
    inv = pd.read_csv(MANUAL / "cpi_consensus.csv", parse_dates=["release_date"])
    inv = inv[inv["release_date"] > j["date"].max()]
    d = pd.concat([j[["date", "actual", "forecast"]].assign(来源="金十"),
                   pd.DataFrame({"date": inv["release_date"], "actual": inv["actual_pct"], "forecast": inv["consensus_pct"], "来源": "Investing"})],
                  ignore_index=True)
    raw = d["actual"] - d["forecast"]
    d["surprise"] = raw / raw.expanding(MIN_HIST).std().shift(1)
    d["时间"] = d["date"] + pd.Timedelta(hours=9, minutes=30)
    return d.dropna(subset=["surprise"]).reset_index(drop=True)


def main() -> None:
    r = reactions(releases(), markets.daily())
    # 核对：金十那部分和第一轮的事件表应当一样
    r1 = pd.read_csv(ROUND1_EVENTS, parse_dates=["date"])
    r1 = r1[r1["indicator"] == "CPI"].set_index("date")
    m = r.set_index("date").join(r1[["surprise", "Δ10年(bp)"]], rsuffix="_第一轮", how="inner")
    print(f"和第一轮对得上的事件 {len(m)} / {len(r1)}；意外最大差 {(m['surprise'] - m['surprise_第一轮']).abs().max():.2g}，"
          f"Δ10年最大差 {(m['Δ10年(bp)'] - m['Δ10年(bp)_第一轮']).abs().max():.2g} bp")
    rng = np.random.default_rng(0)
    rows = []
    for period, base in (("原来（到 2025-08）", r[r["来源"] == "金十"]), (f"延长到 {r['date'].max():%Y-%m}", r)):
        covid = base["时间"].between("2020-01-01", "2021-12-31")
        for label, sub in (("全样本", base), ("截尾 ±3", base.assign(surprise=base["surprise"].clip(-3, 3))), ("去掉 2020–2021", base[~covid])):
            for row in single(sub, rng):
                rows.append({"样本": period, "口径": label, **row})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "surprise_cpi_延长.csv", index=False, encoding="utf-8-sig")
    r.to_csv(RESULTS / "surprise_cpi_延长_events.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(f"新接上的 {int((r['来源'] == 'Investing').sum())} 次：{r.loc[r['来源'] == 'Investing', 'date'].min():%Y-%m-%d} ~ {r['date'].max():%Y-%m-%d}")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
