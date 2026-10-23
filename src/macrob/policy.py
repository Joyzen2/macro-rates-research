"""第 6 步：央行政策公告的事件研究。运行：uv run python -m macrob.policy

规则（看结果之前定的）：
- 事件：降准 / 升准（公布日）；基准贷款利率调整（生效日前一天视为公告日，2006–2015）；1 年期 LPR 变动（报价日，2019-08 起）。
- 窗口：公告日前最后一个交易日收盘 → 公告日之后第一个交易日收盘。盘中、收盘后、周末公布都能盖住。
- 反应：2 年、10 年国债收益率变化（基点），沪深300 涨跌（%）。
- 显著性：同样长度的窗口放在随机的非事件日上（避开事件前后 10 个交易日），同样数量，重复 2000 次，
  看真实平均反应在随机分布里的位置（双侧 p 值）。
- 预期到没有：公告前 20 个交易日 2 年期收益率已经朝政策方向走了 10 个基点以上，算「已有预期」。
修正（看完第一版结果后发现）：同一天宣布的同类事件合并成一次；「全部宽松 / 全部收紧」里同一天的事件只算一次。
修正前的汇总表保留在 results/policy_event_study_修正前.csv。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import markets, sources as S
from .datalayer import RESULTS

N_PERM, ANTICIP_BP, PRE = 2000, 10, 20


def events() -> pd.DataFrame:
    rows = []
    rrr = S.em("macro_china_reserve_requirement_ratio")
    for _, r in rrr.iterrows():
        d = pd.to_datetime(str(r["公布时间"]).replace("年", "-").replace("月", "-").replace("日", ""))
        size = float(r["大型金融机构-调整幅度"])
        if size != 0 and not np.isnan(size):
            rows.append({"date": d, "type": "降准" if size < 0 else "升准", "direction": int(np.sign(size)), "size": size})
    lpr = S.em("macro_china_lpr")
    lpr["TRADE_DATE"] = pd.to_datetime(lpr["TRADE_DATE"])
    b = lpr.dropna(subset=["RATE_1"]).drop_duplicates("RATE_1", keep="first")
    b = lpr.dropna(subset=["RATE_1"]).sort_values("TRADE_DATE")
    b = b[b["RATE_1"].diff().fillna(0) != 0]
    for _, r in b.iterrows():
        if r["TRADE_DATE"] < pd.Timestamp("2006-01-01"):
            continue
        prev = lpr[(lpr["TRADE_DATE"] < r["TRADE_DATE"]) & lpr["RATE_1"].notna()]["RATE_1"].iloc[-1]
        size = float(r["RATE_1"] - prev)
        rows.append({"date": r["TRADE_DATE"] - pd.Timedelta(days=1), "type": "降息（基准）" if size < 0 else "加息（基准）",
                     "direction": int(np.sign(size)), "size": size})
    l1 = lpr.dropna(subset=["LPR1Y"]).sort_values("TRADE_DATE")
    l1 = l1[l1["TRADE_DATE"] >= "2019-08-20"]
    ch = l1[l1["LPR1Y"].diff() != 0].iloc[1:]
    for _, r in ch.iterrows():
        prev = l1[l1["TRADE_DATE"] < r["TRADE_DATE"]]["LPR1Y"].iloc[-1]
        size = float(r["LPR1Y"] - prev)
        rows.append({"date": r["TRADE_DATE"], "type": "LPR 下调" if size < 0 else "LPR 上调", "direction": int(np.sign(size)), "size": size})
    e = pd.DataFrame(rows)
    # 修正（看完第一版结果后发现）：同一天宣布、分两步实施的降准（2019-01-04）原来算成了两次，合并成一次
    e = e.groupby(["date", "type", "direction"], as_index=False)["size"].sum().sort_values("date").reset_index(drop=True)
    return e[e["date"] >= "2006-01-01"]


def _window(days: pd.DatetimeIndex, d: pd.Timestamp) -> tuple[int, int] | None:
    before = days.searchsorted(d, side="left") - 1           # 公告日之前最后一个交易日
    after = days.searchsorted(d, side="right")               # 公告日之后第一个交易日
    if before < PRE or after >= len(days):
        return None
    return before, after


def reactions(e: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    days, out = mk.index, []
    for _, r in e.iterrows():
        w = _window(days, r["date"])
        if w is None:
            continue
        a, b = w
        pre = (mk["y2"].iloc[a] - mk["y2"].iloc[a - PRE]) * 100
        out.append({**r.to_dict(), "Δ2年(bp)": (mk["y2"].iloc[b] - mk["y2"].iloc[a]) * 100,
                    "Δ10年(bp)": (mk["y10"].iloc[b] - mk["y10"].iloc[a]) * 100,
                    "沪深300(%)": (mk["hs300"].iloc[b] / mk["hs300"].iloc[a] - 1) * 100,
                    "公告前20日Δ2年(bp)": pre, "窗口交易日数": b - a})
    out = pd.DataFrame(out)
    out["已有预期"] = (np.sign(out["公告前20日Δ2年(bp)"]) == out["direction"]) & (out["公告前20日Δ2年(bp)"].abs() >= ANTICIP_BP)
    return out


def placebo_p(real: pd.DataFrame, mk: pd.DataFrame, col: str, rng) -> float:
    days = mk.index
    event_pos = set()
    for d in real["date"]:
        p = days.searchsorted(d)
        event_pos.update(range(p - 10, p + 11))
    cand = np.array([i for i in range(PRE + 1, len(days) - 3) if i not in event_pos])
    width = int(round(real["窗口交易日数"].mean()))
    series = {"Δ2年(bp)": mk["y2"].to_numpy() * 100, "Δ10年(bp)": mk["y10"].to_numpy() * 100, "沪深300(%)": np.log(mk["hs300"].to_numpy()) * 100}[col]
    target = real[col].mean()
    fake = []
    for _ in range(N_PERM):
        a = rng.choice(cand, size=len(real), replace=False)
        diff = series[a + width] - series[a]
        fake.append(np.nanmean(diff))
    fake = np.array(fake)
    return float((np.abs(fake - fake.mean()) >= abs(target - fake.mean())).mean())


def main() -> None:
    mk = markets.daily()
    e = events()
    r = reactions(e, mk)
    r.to_csv(RESULTS / "policy_events.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(0)
    rows = []
    # 修正：同一天既降准又降息的（2015-08-25、2015-10-23），在「全部宽松 / 全部收紧」里只算一次
    ease, tight = (r[r["direction"] == k].drop_duplicates("date") for k in (-1, 1))
    for (t, grp) in list(r.groupby("type")) + [("全部宽松", ease), ("全部收紧", tight)]:
        row = {"事件": t, "次数": len(grp), "起": grp["date"].min().strftime("%Y-%m"), "止": grp["date"].max().strftime("%Y-%m")}
        for col in ("Δ2年(bp)", "Δ10年(bp)", "沪深300(%)"):
            x = grp[col].dropna()
            row[col] = x.mean()
            row[col + " t"] = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan
            row[col + " 安慰剂p"] = placebo_p(grp.dropna(subset=[col]), mk, col, rng) if len(x) > 2 else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "policy_event_study.csv", index=False, encoding="utf-8-sig")
    ant = ease.groupby("已有预期")[["Δ2年(bp)", "Δ10年(bp)", "沪深300(%)"]].agg(["mean", "count"])
    ant.to_csv(RESULTS / "policy_anticipation.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print(t.round(3).to_string(index=False))
    print("\n宽松事件按「已有预期」分组：\n", ant.round(2).to_string())


if __name__ == "__main__":
    main()
