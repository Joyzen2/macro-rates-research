"""补做：7 天逆回购和 MLF 调息的事件研究。运行：uv run python -m macrob.policy_rates

规则（看结果之前定的，写在计划文件第 6 步「补做」里）：
- 事件：data/policy_rate_changes.csv（逐次用新闻报道核实日期），同一天调的合并成一个事件；
  剔除 2018-04-17（MLF 上调和降准同一天宣布，方向相反）；两次新闻发布会（2024-09-24、2025-05-07）单列。
- 分组：前 30 天内没有同方向的 7 天逆回购或 MLF 调整 = 首发，否则 = 跟随。
  第 6 步的 LPR 下调按同样规则分跟随 / 单独；和 7 天逆回购同一天的 LPR（2024-07-22）只算在 7 天逆回购里。
- 加息事件的反应乘以 −1，统一换算成「宽松方向」；另报只看降息的。
- 窗口、反应和第 6 步一样；安慰剂：随机窗口避开所有政策事件前后 10 个交易日，
  按真实事件的方向顺序取号，重复 2000 次，双侧 p 值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import markets, policy
from .datalayer import RESULTS, ROOT

N_PERM, LOOKBACK, AVOID = 2000, 30, 10
EXCLUDE = [pd.Timestamp("2018-04-17")]
COLS = ("Δ2年(bp)", "Δ10年(bp)", "沪深300(%)")


def rate_events() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "policy_rate_changes.csv", parse_dates=["date"])
    d = d[~d["date"].isin(EXCLUDE)]
    rows = []
    for date, g in d.groupby("date"):
        main = g[g["instrument"] == "7天逆回购"]
        size = float((main if len(main) else g)["change_bp"].iloc[0])
        rows.append({"date": date, "工具": "+".join(g["instrument"]), "size": size, "direction": int(np.sign(size))})
    e = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    group = []
    for _, r in e.iterrows():
        if r["工具"] == "发布会":
            group.append("发布会")
            continue
        prior = e[(e["date"] < r["date"]) & (e["date"] >= r["date"] - pd.Timedelta(days=LOOKBACK)) & (e["direction"] == r["direction"])]
        group.append("跟随" if len(prior) else "首发")
    e["分组"] = group
    return e


def lpr_events(rates: pd.DataFrame) -> pd.DataFrame:
    old = policy.events()
    lpr = old[old["type"] == "LPR 下调"].copy()
    lpr = lpr[~lpr["date"].isin(rates["date"])]                          # 2024-07-22 只算在 7 天逆回购里
    group = []
    for _, r in lpr.iterrows():
        prior = rates[(rates["date"] < r["date"]) & (rates["date"] >= r["date"] - pd.Timedelta(days=LOOKBACK)) & (rates["direction"] == r["direction"])]
        group.append("LPR 跟随" if len(prior) else "LPR 单独")
    lpr["分组"] = group
    lpr["工具"] = "LPR"
    return lpr[["date", "工具", "size", "direction", "分组"]]


def _avoid_positions(days: pd.DatetimeIndex) -> set:
    all_dates = pd.concat([policy.events()["date"], rate_events()["date"]])
    out = set()
    for d in all_dates:
        p = days.searchsorted(d)
        out.update(range(p - AVOID, p + AVOID + 1))
    return out


def summarize(r: pd.DataFrame, mk: pd.DataFrame, rng) -> pd.DataFrame:
    days = mk.index
    avoid = _avoid_positions(days)
    cand = np.array([i for i in range(policy.PRE + 1, len(days) - 3) if i not in avoid])
    level = {"Δ2年(bp)": mk["y2"].to_numpy() * 100, "Δ10年(bp)": mk["y10"].to_numpy() * 100, "沪深300(%)": mk["hs300"].to_numpy()}
    groups = [("首发（加息乘 −1）", r[r["分组"] == "首发"]), ("首发·只看降息", r[(r["分组"] == "首发") & (r["direction"] < 0)]),
              ("跟随（7 天逆回购 / MLF）", r[r["分组"] == "跟随"]), ("LPR 跟随", r[r["分组"] == "LPR 跟随"]),
              ("LPR 单独", r[r["分组"] == "LPR 单独"]), ("发布会", r[r["分组"] == "发布会"])]
    rows = []
    for name, g in groups:
        row = {"分组": name, "次数": len(g), "其中加息": int((g["direction"] > 0).sum()), "日期": "、".join(g["date"].dt.strftime("%Y-%m-%d"))}
        sign = -g["direction"].to_numpy()
        width = 2
        for col in COLS:
            x = g[col].to_numpy() * sign
            row[col] = x.mean()
            row[col + " t"] = x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan
            if len(x) > 1:
                fake = []
                for _ in range(N_PERM):
                    a = rng.choice(cand, size=len(g), replace=False)
                    s = level[col]
                    diff = (s[a + width] / s[a] - 1) * 100 if col == "沪深300(%)" else s[a + width] - s[a]
                    fake.append(np.nanmean(diff * sign))
                fake = np.array(fake)
                row[col + " 安慰剂p"] = float((np.abs(fake - fake.mean()) >= abs(x.mean() - fake.mean())).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    mk = markets.daily()
    rates = rate_events()
    e = pd.concat([rates, lpr_events(rates)], ignore_index=True).sort_values("date").reset_index(drop=True)
    e["type"] = e["分组"]
    r = policy.reactions(e, mk)
    r.drop(columns=["type", "已有预期", "公告前20日Δ2年(bp)"]).to_csv(RESULTS / "policy_rate_events.csv", index=False, encoding="utf-8-sig")
    t = summarize(r, mk, np.random.default_rng(0))
    t.to_csv(RESULTS / "policy_rate_event_study.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print(r[["date", "工具", "分组", "size", "Δ2年(bp)", "Δ10年(bp)", "沪深300(%)"]].round(2).to_string(index=False))
    print()
    print(t.drop(columns=["日期"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
