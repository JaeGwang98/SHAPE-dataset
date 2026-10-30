# -*- coding: utf-8 -*-
"""모델별 이질성 ablation용 학습셋 3종 생성 (모두 78,000, 라벨 균형).
  pm_4o.csv   : 순수 GPT-4o-mini 78,000
  pm_41.csv   : 순수 GPT-4.1-mini 78,000
  pm_mix.csv  : 39,000(4o) + 39,000(4.1) = 78,000 (50:50, 라벨당 1,500+1,500)
크기를 고정해 '혼합 여부'만 변수로 격리한다."""
import pandas as pd
D = "../2025_05_19_data/사용할 데이터"
OUT = "results"
def load_model(prefix):
    a = pd.read_csv(f"{D}/{prefix}_rest.csv")
    b = pd.read_csv(f"{D}/{prefix}_origin_5200.csv")
    df = pd.concat([a, b], ignore_index=True)[["질문", "labels"]]
    return df
o4 = load_model("4o_mini")
m41 = load_model("4.1_mini")
print("4o:", len(o4), "4.1:", len(m41), "labels:", o4.labels.nunique(), m41.labels.nunique())
o4.to_csv(f"{OUT}/pm_4o.csv", index=False, encoding="utf-8-sig")
m41.to_csv(f"{OUT}/pm_41.csv", index=False, encoding="utf-8-sig")
# 혼합: 라벨당 1500씩 각 모델에서
def half(df, seed):
    return df.groupby("labels", group_keys=False).apply(
        lambda g: g.sample(n=min(1500, len(g)), random_state=seed)).reset_index(drop=True)
mix = pd.concat([half(o4, 42), half(m41, 43)], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
mix.to_csv(f"{OUT}/pm_mix.csv", index=False, encoding="utf-8-sig")
print("pm_4o:", len(o4), "pm_41:", len(m41), "pm_mix:", len(mix))
