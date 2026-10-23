"""第 12 步（补数据）：社零、固投的数据意外，一致预期来自华尔街见闻财经日历。运行：uv run python -m macrob.surprise_nbs

规则（看结果之前定的，写在计划文件第 12 步里）：
- 意外 = 公布值 − 预期值，除以之前所有发布的意外标准差（至少 12 次），只用过去的数据。
- 统计局 10 点发布（在交易时段内）：窗口 = 发布日前一个交易日收盘 → 发布日收盘；发布日不是交易日或在收盘后发布，顺延到下一个交易日。
- 社零用当月同比（1–2 月用合并值），固投用累计同比；工业增加值同口径，用来和第一轮金十的结果对照。
- 每个指标单独回归（异方差稳健 t + 打乱检验 2000 次）；三个指标同一天发布，另做一个联合回归。
看完结果后加的稳健性（标明是修正）：全样本被 2020-03-16 那一次主导（固投意外 −49 个标准差），
所以另报「去掉 2020–2021 年」；并用统计局的数核对公布值，再去掉对不上的（比如 2011-03 社零记的是实际增速）。
补数据（续）：华尔街见闻 2024 年起没有预期，改用本人找来的 Investing 预期（data/manual/），延长到 2026-09，结果另存「_延长」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import markets
from .datalayer import RESULTS, raw_panel
from .probe_util import RAW

MIN_HIST, N_PERM = 12, 2000
MANUAL = RAW.parent / "manual"                                   # 本人找来的数据
INVESTING = {"社零": "retail_consensus.csv", "固投": "fai_consensus.csv"}
EXTEND_FROM = pd.Timestamp("2024-01-01")
PATTERNS = {
    "工业增加值": r"^(\d+月|1至2月)规模以上工业增加值同比$",
    "社零": r"^(\d+月|1至2月)社会消费品零售总额同比$",
    "固投": r"^1至\d+月城镇固定资产投资同比$",
}
COLS = ("Δ10年(bp)", "Δ2年(bp)", "沪深300(%)")
NBS_KEY = {"工业增加值": "ip_yoy", "社零": "retail_yoy", "固投": "fai_ytd"}


def releases(name: str, extend: bool = False) -> pd.DataFrame:
    d = pd.read_csv(RAW / "wscn_calendar_china.csv", parse_dates=["时间"])
    d = d[d["title"].str.match(PATTERNS[name])]
    d = d.assign(actual=pd.to_numeric(d["actual"], errors="coerce"), forecast=pd.to_numeric(d["forecast"], errors="coerce"))
    if extend and name in INVESTING:                             # 2024 年起的预期用 Investing，公布值和时刻还用华尔街见闻
        inv = pd.read_csv(MANUAL / INVESTING[name], parse_dates=["release_date"]).set_index("release_date")["consensus_pct"]
        late = d["时间"] >= EXTEND_FROM
        d.loc[late, "forecast"] = d.loc[late, "时间"].dt.normalize().map(inv)
    d = d.dropna(subset=["actual", "forecast"]).sort_values("时间").drop_duplicates("时间", keep="first")
    raw = d["actual"] - d["forecast"]
    d["surprise"] = raw / raw.expanding(MIN_HIST).std().shift(1)
    # 核对：标题里的月份 → 数据所属月份（12 月的数次年 1 月发布），和统计局的数对比
    mon = d["title"].str.extract(r"至?(\d+)月")[0].astype(int)
    year = d["时间"].dt.year - (mon > d["时间"].dt.month).astype(int)
    ref = pd.to_datetime(dict(year=year, month=mon, day=1)) + pd.offsets.MonthEnd(0)
    nbs = raw_panel()[NBS_KEY[name]]
    d["统计局的数"] = nbs.reindex(ref).to_numpy()
    diff = (d["actual"] - d["统计局的数"]).abs()
    d["和统计局一致"] = np.where(diff.isna(), "没法核对", np.where(diff <= 0.15, "一致", "对不上"))   # 1–2 月合并值统计局库里没有
    return d.dropna(subset=["surprise"])[["时间", "title", "actual", "forecast", "surprise", "统计局的数", "和统计局一致"]].reset_index(drop=True)


def windows(times: pd.Series, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    date = times.dt.normalize()
    before = days.searchsorted(date, side="left") - 1
    same_day = days.searchsorted(date, side="left")
    in_session = (same_day < len(days)) & (days[np.minimum(same_day, len(days) - 1)] == date) & (times.dt.hour < 15)
    after = np.where(in_session, same_day, days.searchsorted(date, side="right"))
    return before, after


def reactions(ev: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    b, a = windows(ev["时间"], mk.index)
    ok = (b >= 0) & (a < len(mk))
    ev, b, a = ev[ok].reset_index(drop=True), b[ok], a[ok]
    A, B = mk.iloc[a].reset_index(drop=True), mk.iloc[b].reset_index(drop=True)
    return ev.assign(**{"Δ10年(bp)": (A["y10"] - B["y10"]) * 100, "Δ2年(bp)": (A["y2"] - B["y2"]) * 100,
                        "沪深300(%)": (A["hs300"] / B["hs300"] - 1) * 100, "事件日": mk.index[a]})


def single(r: pd.DataFrame, rng) -> list[dict]:
    rows = []
    x = r["surprise"].to_numpy()
    for col in COLS:
        y = r[col].to_numpy()
        m = ~np.isnan(y)
        fit = sm.OLS(y[m], sm.add_constant(x[m])).fit(cov_type="HC1")
        perm = np.array([np.polyfit(rng.permutation(x[m]), y[m], 1)[0] for _ in range(N_PERM)])
        rows.append({"资产": col, "每 1 个标准差的反应": float(fit.params[1]), "t": float(fit.tvalues[1]),
                     "打乱 p": float((np.abs(perm) >= abs(fit.params[1])).mean()), "事件数": int(m.sum())})
    return rows


def run(extend: bool, suffix: str) -> None:
    mk = markets.daily()
    rng = np.random.default_rng(0)
    out, per = [], {}
    for name in PATTERNS:
        ev = releases(name, extend)
        r = reactions(ev, mk)
        per[name] = r
        for row in single(r, rng):
            out.append({"指标": name, "起": f"{r['时间'].min():%Y-%m}", "止": f"{r['时间'].max():%Y-%m}", **row})
    t = pd.DataFrame(out)
    # 看完结果后加的稳健性：去掉 2020–2021；再去掉和统计局对不上的
    rob = []
    for name, r in per.items():
        for label, sub in (("去掉 2020–2021", r[~r["时间"].between("2020-01-01", "2021-12-31")]),
                           ("再去掉和统计局对不上的", r[~r["时间"].between("2020-01-01", "2021-12-31") & (r["和统计局一致"] != "对不上")])):
            for row in single(sub, rng):
                rob.append({"指标": name, "样本": label, "核对一致": int((r["和统计局一致"] == "一致").sum()),
                            "对不上": int((r["和统计局一致"] == "对不上").sum()), "没法核对": int((r["和统计局一致"] == "没法核对").sum()), **row})
    rt = pd.DataFrame(rob)
    rt.to_csv(RESULTS / f"surprise_nbs_robust{suffix}.csv", index=False, encoding="utf-8-sig")
    # 联合回归：三个指标同一个事件日
    j = None
    for name, r in per.items():
        s = r.groupby("事件日").agg(**{name: ("surprise", "mean")})
        j = s if j is None else j.join(s, how="inner")
    y_all = per["工业增加值"].groupby("事件日")[list(COLS)].first().reindex(j.index)
    joint = []
    for col in COLS:
        d = pd.concat([j, y_all[col]], axis=1).dropna()
        fit = sm.OLS(d[col], sm.add_constant(d[list(PATTERNS)])).fit(cov_type="HC1")
        for name in PATTERNS:
            joint.append({"资产": col, "指标": name, "系数": float(fit.params[name]), "t": float(fit.tvalues[name]), "事件日数": len(d)})
    jt = pd.DataFrame(joint)
    t.to_csv(RESULTS / f"surprise_nbs_single{suffix}.csv", index=False, encoding="utf-8-sig")
    jt.to_csv(RESULTS / f"surprise_nbs_joint{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.concat(per.values(), keys=per.keys(), names=["指标"]).to_csv(RESULTS / f"surprise_nbs_events{suffix}.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"===== {'延长（2024 年起用 Investing 的预期）' if extend else '事先定的样本'}")
    print(t.round(3).to_string(index=False))
    print("\n联合回归（同一天发布的三个意外一起放）：")
    print(jt.round(3).to_string(index=False))
    print("\n看完结果后加的稳健性：")
    print(rt.round(3).to_string(index=False))


def main() -> None:
    run(False, "")
    run(True, "_延长")


if __name__ == "__main__":
    main()
