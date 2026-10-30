# -*- coding: utf-8 -*-
"""
Reviewer 3-1 대응: 26개 질병 클래스 간 증상 유사도 정량화.
근거 데이터: Final_data_0513.csv (전처리된 27,601행, 381개 증상 이진행렬)
산출물:
  - results/class_symptom_similarity.csv   (26x26 코사인 유사도 행렬)
  - results/class_symptom_jaccard.csv      (26x26 자카드 유사도 행렬)
  - results/most_confusable_pairs.csv      (유사도 상위 질병쌍)
  - results/class_similarity_heatmap.png   (히트맵)
"""
import pandas as pd, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "../Final_data_0513.csv"
OUT = "results"

df = pd.read_csv(SRC)
meta = ["eng_name", "Kor_name", "Major", "diseases"]
symptom_cols = [c for c in df.columns if c not in meta]
# 숫자형(이진)만 유지
X = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
X = X.loc[:, (X.isin([0, 1]).all())]      # 순수 이진 컬럼만
print(f"이진 증상 컬럼: {X.shape[1]}개, 행: {len(X)}")

df2 = pd.concat([df["Kor_name"], X], axis=1)
# 클래스별 평균 증상 프로파일(증상 출현 빈도)
proto = df2.groupby("Kor_name")[X.columns].mean()
classes = list(proto.index)
print(f"클래스: {len(classes)}개")
# 한글->영문 질병명 매핑(영문 라벨 히트맵용)
kor2eng = (df.dropna(subset=["Kor_name", "eng_name"])
             .drop_duplicates("Kor_name").set_index("Kor_name")["eng_name"].to_dict())
eng_labels = [str(kor2eng.get(c, c)) for c in classes]

# 코사인 유사도
cos = cosine_similarity(proto.values)
cos_df = pd.DataFrame(cos, index=classes, columns=classes)
cos_df.to_csv(f"{OUT}/class_symptom_similarity.csv", encoding="utf-8-sig")

# 자카드 유사도(이진화한 프로토타입: 해당 클래스에서 한 번이라도 등장한 증상)
binp = (proto.values > 0).astype(int)
inter = binp @ binp.T
row = binp.sum(1, keepdims=True)
union = row + row.T - inter
jac = inter / np.maximum(union, 1)
jac_df = pd.DataFrame(jac, index=classes, columns=classes)
jac_df.to_csv(f"{OUT}/class_symptom_jaccard.csv", encoding="utf-8-sig")

# 가장 혼동되기 쉬운 질병쌍 (코사인 상위)
pairs = []
for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        pairs.append((classes[i], classes[j], cos[i, j], jac[i, j]))
pairs.sort(key=lambda x: -x[2])
pair_df = pd.DataFrame(pairs, columns=["disease_A", "disease_B", "cosine", "jaccard"])
pair_df.to_csv(f"{OUT}/most_confusable_pairs.csv", index=False, encoding="utf-8-sig")

print("\n=== 증상 유사도 상위 10개 질병쌍 (혼동 가능성 높음) ===")
print(pair_df.head(10).to_string(index=False))

# 요약 통계
offdiag = cos[~np.eye(len(classes), dtype=bool)]
print(f"\n클래스 간 코사인 유사도: 평균={offdiag.mean():.3f}, 최대={offdiag.max():.3f}, "
      f"상위5%={np.percentile(offdiag,95):.3f}")

# 히트맵
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cos, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(eng_labels, rotation=90, fontsize=7)
ax.set_yticklabels(eng_labels, fontsize=7)
ax.set_title("Inter-class symptom-profile cosine similarity (26 diseases)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{OUT}/class_similarity_heatmap.png", dpi=200)
print(f"\n저장 완료: {OUT}/ (similarity csv 2종, confusable pairs, heatmap)")
