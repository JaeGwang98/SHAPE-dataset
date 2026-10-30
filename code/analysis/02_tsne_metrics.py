# -*- coding: utf-8 -*-
"""
Reviewer 3-2 대응: t-SNE를 '탐색적 시각화'로 위상 낮추고, 정량 지표로 보강.
- 생성 문장을 Sentence-BERT로 임베딩 후:
    * silhouette score      (클래스 간 분리도)
    * kNN label agreement   (클래스 내 응집도 = 최근접이웃 라벨 일치율)
- 처리전 이진 증상행렬(processed data)에 대해서도 동일 지표 산출(Fig 4a 대응)
산출물: results/tsne_quant_metrics.json, results/tsne_generated.png
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA = "../2025_05_19_data/사용할 데이터"
TOTAL = f"{DATA}/25.06.04/total_dataset.csv"
BASE = "../Final_data_0513.csv"
PER_CLASS = 300
SEED = 42
OUT = "results"

def knn_agreement(X, y, k=10):
    knn = KNeighborsClassifier(n_neighbors=k)
    pred = cross_val_predict(knn, X, y, cv=3, n_jobs=-1)
    return float((pred == y).mean())

# ---------- 1) 생성 문장 임베딩 ----------
df = pd.read_csv(TOTAL)
samp = df.groupby("labels", group_keys=False).apply(
    lambda g: g.sample(n=min(PER_CLASS, len(g)), random_state=SEED)).reset_index(drop=True)
print(f"생성 문장 표본: {len(samp)} ({samp['labels'].nunique()} classes)")

from sentence_transformers import SentenceTransformer
model_name = "jhgan/ko-sroberta-multitask"
try:
    st = SentenceTransformer(model_name)
except Exception as e:
    print("fallback multilingual:", e)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    st = SentenceTransformer(model_name)
emb = st.encode(samp["질문"].astype(str).tolist(), batch_size=128,
                show_progress_bar=False, convert_to_numpy=True)
ylab = samp["labels"].values

sil_gen = silhouette_score(emb, ylab, sample_size=5000, random_state=SEED)
knn_gen = knn_agreement(emb, ylab, k=10)
print(f"[생성문장/{model_name}] silhouette={sil_gen:.4f}  kNN10={knn_gen:.4f}")

# ---------- 2) 처리전 증상 이진행렬 ----------
base = pd.read_csv(BASE)
meta = ["eng_name", "Kor_name", "Major", "diseases"]
sym = base[[c for c in base.columns if c not in meta]].apply(
    pd.to_numeric, errors="coerce").fillna(0)
sym = sym.loc[:, sym.isin([0, 1]).all()]
bsamp = pd.concat([base["Kor_name"], sym], axis=1).groupby("Kor_name", group_keys=False).apply(
    lambda g: g.sample(n=min(PER_CLASS, len(g)), random_state=SEED)).reset_index(drop=True)
Xb = bsamp[sym.columns].values
yb = bsamp["Kor_name"].values
Xb50 = PCA(n_components=50, random_state=SEED).fit_transform(Xb)
sil_base = silhouette_score(Xb50, yb, sample_size=5000, random_state=SEED)
knn_base = knn_agreement(Xb50, yb, k=10)
print(f"[처리전 증상행렬] silhouette={sil_base:.4f}  kNN10={knn_base:.4f}")

# ---------- 3) t-SNE 시각화(생성문장) ----------
tsne = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30)
Z = tsne.fit_transform(emb)
kor2eng = (base.dropna(subset=["Kor_name","eng_name"]).drop_duplicates("Kor_name")
           .set_index("Kor_name")["eng_name"].to_dict())
fig, ax = plt.subplots(figsize=(11, 9))
classes = sorted(pd.unique(ylab))
cmap = plt.get_cmap("tab20", len(classes))
for i, c in enumerate(classes):
    m = ylab == c
    ax.scatter(Z[m, 0], Z[m, 1], s=6, color=cmap(i),
               label=str(kor2eng.get(c, c)), alpha=0.6)
ax.set_title(f"t-SNE of SHAPE sentences (Sentence-BERT)\n"
             f"silhouette={sil_gen:.3f}, kNN-10 agreement={knn_gen:.3f}")
ax.legend(fontsize=6, ncol=2, markerscale=2, loc="upper right", bbox_to_anchor=(1.16, 1))
plt.tight_layout()
plt.savefig(f"{OUT}/tsne_generated.png", dpi=200, bbox_inches="tight")

res = {"embed_model": model_name, "per_class": PER_CLASS,
       "generated_silhouette": float(sil_gen), "generated_knn10_agreement": float(knn_gen),
       "processed_symptom_silhouette": float(sil_base),
       "processed_symptom_knn10_agreement": float(knn_base),
       "n_generated": int(len(samp)), "n_processed": int(len(bsamp))}
json.dump(res, open(f"{OUT}/tsne_quant_metrics.json", "w"), ensure_ascii=False, indent=2)
print("\n저장:", f"{OUT}/tsne_quant_metrics.json, tsne_generated.png")
