# -*- coding: utf-8 -*-
"""
2차 리뷰(R6-2, R8-5, R9-2) 대응: SHAPE 고유의 RI(결측 증상 1개 삽입 후 LLM 생성) 효과 검증.
근거: Test.py 생성 규칙 — 원본 행마다 5개 프롬프트(i=0: RS-only, i=1..4: RS + 결측 증상 1개 삽입).
      *_total.csv 는 질병별 3000행 블록이며 행 인덱스 % 5 == 0 이 RS-only.
      (data_separate.py 가 각 블록의 0,5,...,995 를 Testset 으로 분리 -> Testset 은 전부 RS-only)
산출:
  results/ri_flags_trainset.csv : Trainset 각 행의 variant(0=RS-only, 1..4=RS+RI), source(4o-mini/4.1-mini)
  results/ri_rsonly.csv  : Trainset 의 RS-only 전체 (질병당 800 = 2모델 x 400)
  results/ri_rsri.csv    : RS-only : RS+RI = 1:4 자연비율, 질병당 800 (크기 일치)
  results/ri_rionly.csv  : RS+RI 만, 질병당 800 (크기 일치)
"""
import pandas as pd
D = "../2025_05_19_data"
U = f"{D}/사용할 데이터"
SEED = 42

def flags(total_csv, src):
    t = pd.read_csv(total_csv)
    assert len(t) == 78000
    t["variant"] = t.index % 5
    t["source"] = src
    return t

tot = pd.concat([flags(f"{D}/4o_mini_total.csv", "4o-mini"),
                 flags(f"{D}/4.1_mini_total.csv", "4.1-mini")], ignore_index=True)
tr = pd.read_csv(f"{U}/25.05.19/Trainset.csv")
te = pd.read_csv(f"{U}/25.05.19/Testset.csv")
assert not tot.duplicated(["질문", "source"]).any()
# 텍스트 중복(두 모델이 같은 문장을 낸 경우)은 첫 등장 기준
key = tot.drop_duplicates("질문").set_index("질문")
m = tr.join(key[["variant", "source"]], on="질문")
print("unmatched train rows:", m["variant"].isna().sum())
mt = te.join(key[["variant", "source"]], on="질문")
print("Testset variant counts (NaN = GPT-4o rows):", mt["variant"].value_counts(dropna=False).to_dict())
m.to_csv("results/ri_flags_trainset.csv", index=False, encoding="utf-8-sig")
print("Trainset variant counts:", m["variant"].value_counts().sort_index().to_dict())

rs = m[m["variant"] == 0]; ri = m[m["variant"] > 0]
n = rs.groupby("labels").size().min()
print("RS-only per class (min):", n)
def samp(df, k, seed):
    return df.groupby("labels", group_keys=False).apply(lambda g: g.sample(n=k, random_state=seed))
arm_rs = samp(rs, n, SEED)
arm_rsri = pd.concat([samp(rs, n // 5, SEED + 1), samp(ri, n - n // 5, SEED + 1)])
arm_ri = samp(ri, n, SEED + 2)
for name, a in [("ri_rsonly", arm_rs), ("ri_rsri", arm_rsri), ("ri_rionly", arm_ri)]:
    a[["질문", "labels"]].sample(frac=1, random_state=SEED).to_csv(
        f"results/{name}.csv", index=False, encoding="utf-8-sig")
    print(name, len(a), a["variant"].gt(0).mean().round(3))
