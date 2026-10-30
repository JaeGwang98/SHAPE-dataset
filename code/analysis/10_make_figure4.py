# -*- coding: utf-8 -*-
"""
Reviewer 1 대응: Figure 4 재작성.
지적 사항 — "Figures require clearer axis labels and comprehensive legends."

원본 Figure 4는 축 라벨·범례·클래스 색상 키가 없었다. 이 스크립트는 동일한 데이터로
두 패널을 다시 그리되 다음을 갖춘다.
  * 축 라벨 (t-SNE dimension 1 / 2)
  * 장기 계통 5군 범례 + 군마다 다른 마커 형태 (색각 이상 대비 보조 부호화)
  * 26개 질환 전부에 직접 텍스트 라벨 (색상만으로 식별하지 않게 함)
  * 패널별 정량 지표 주석 (silhouette, kNN-10) — 리뷰어 3 대응과 연결

산출물: results/Figure4.png / Figure4.pdf / Figure4a.png / Figure4b.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA = "../2025_05_19_data/사용할 데이터"
TOTAL = f"{DATA}/25.06.04/total_dataset.csv"
BASE = "../Final_data_0513.csv"
OUT = "results"
PER_CLASS = 300
SEED = 42

# 본문 Table 4 표기와 일치시킨 영문 질환명
KOR2ENG = {
    "외이염": "Otitis externa", "중이염": "Otitis media", "결막염": "Conjunctivitis",
    "안검염": "Blepharitis", "인후두염": "Laryngopharyngitis", "부비동염": "Sinusitis",
    "편도염": "Tonsillitis", "골관절염": "Osteoarthritis", "림프절염": "Lymphadenitis",
    "감기": "Common cold", "독감": "Influenza", "방광염": "Cystitis",
    "요도염": "Urethritis", "자궁경부염": "Cervicitis", "전립선염": "Prostatitis",
    "질염": "Vaginitis", "간염": "Hepatitis", "담낭염": "Cholecystitis",
    "위염": "Gastritis", "위장염": "Gastroenteritis", "구내염": "Stomatitis",
    "치주염": "Periodontitis", "접촉성 피부염": "Contact dermatitis",
    "지루성 피부염": "Seborrheic dermatitis", "기관지염": "Bronchitis", "폐렴": "Pneumonia",
}

# 장기 계통 5군 — 색은 validate_palette.js 전체쌍 검증 통과 조합
GROUPS = [
    ("Head and neck",           "#0072B2", "o",
     ["Conjunctivitis", "Blepharitis", "Otitis externa", "Otitis media", "Sinusitis",
      "Tonsillitis", "Laryngopharyngitis", "Stomatitis", "Periodontitis"]),
    ("Respiratory",             "#D55E00", "s",
     ["Bronchitis", "Pneumonia", "Common cold", "Influenza"]),
    ("Gastrointestinal and hepatobiliary", "#009E73", "^",
     ["Gastritis", "Gastroenteritis", "Hepatitis", "Cholecystitis"]),
    ("Genitourinary",           "#CC79A7", "D",
     ["Cystitis", "Urethritis", "Prostatitis", "Cervicitis", "Vaginitis"]),
    ("Skin and musculoskeletal", "#6A3D9A", "v",
     ["Contact dermatitis", "Seborrheic dermatitis", "Osteoarthritis", "Lymphadenitis"]),
]
ENG2GRP = {d: i for i, (_, _, _, ds) in enumerate(GROUPS) for d in ds}


def declutter(xy, texts, iters=900, char_w=0.0092, half_h=0.0165, anchor_pull=0.02):
    """
    겹치는 라벨을 밀어내는 2차원 반발 배치 (adjustText 대체).
    각 라벨을 정규화 축좌표상의 사각형으로 보고, 겹치면 겹침이 적은 축 방향으로 분리한다.
    원래 중심점에서 너무 멀어지지 않도록 매 반복 약하게 되당긴다.
    """
    pos = xy.astype(float).copy()
    anchor = xy.astype(float).copy()
    hw = np.array([max(2.5, len(t)) * char_w for t in texts])  # 라벨 반폭
    for _ in range(iters):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                ox = hw[i] + hw[j] - abs(dx)      # x축 겹침량
                oy = half_h * 2 - abs(dy)          # y축 겹침량
                if ox <= 0 or oy <= 0:
                    continue
                # 겹침이 작은 축으로 분리하는 편이 이동량이 적다
                if oy <= ox * 0.55:
                    s = 1.0 if dy >= 0 else -1.0
                    push = oy / 2 + 1e-4
                    pos[j, 1] += s * push
                    pos[i, 1] -= s * push
                else:
                    s = 1.0 if dx >= 0 else -1.0
                    push = ox / 2 + 1e-4
                    pos[j, 0] += s * push
                    pos[i, 0] -= s * push
                moved = True
        # 앵커로 약하게 되당기기
        pos += (anchor - pos) * anchor_pull
        if not moved:
            break
    return pos


def panel(ax, Z, eng_labels, title, metrics, show_legend):
    order = [g[0] for g in GROUPS]
    for gi, (gname, color, marker, _) in enumerate(GROUPS):
        m = np.array([ENG2GRP.get(e, -1) == gi for e in eng_labels])
        if not m.any():
            continue
        ax.scatter(Z[m, 0], Z[m, 1], s=7, c=color, marker=marker,
                   alpha=0.45, linewidths=0, rasterized=True, zorder=2)

    # 26개 질환 중심점에 직접 라벨 — 색상만으로 식별하지 않도록
    cents, names, cols = [], [], []
    for eng in sorted(set(eng_labels)):
        m = np.array([e == eng for e in eng_labels])
        cents.append([np.median(Z[m, 0]), np.median(Z[m, 1])])
        names.append(eng)
        cols.append(GROUPS[ENG2GRP.get(eng, 0)][1])
    cents = np.array(cents)

    # 축 범위를 정규화한 좌표에서 라벨을 밀어낸 뒤 되돌림
    x0, x1 = Z[:, 0].min(), Z[:, 0].max()
    y0, y1 = Z[:, 1].min(), Z[:, 1].max()
    norm = np.column_stack([(cents[:, 0] - x0) / (x1 - x0), (cents[:, 1] - y0) / (y1 - y0)])
    placed = declutter(norm, names)
    placed[:, 0] = placed[:, 0] * (x1 - x0) + x0
    placed[:, 1] = placed[:, 1] * (y1 - y0) + y0

    for (px, py), (cx, cy), nm, cl in zip(placed, cents, names, cols):
        far = np.hypot((px - cx) / (x1 - x0), (py - cy) / (y1 - y0))
        if far > 0.012:
            ax.plot([cx, px], [cy, py], color=cl, lw=0.5, alpha=0.55, zorder=3)
        ax.text(px, py, nm, fontsize=6.6, color="#1a1a1a", ha="center", va="center",
                zorder=4, path_effects=[pe.withStroke(linewidth=2.1, foreground="white")])

    ax.set_xlabel("t-SNE dimension 1", fontsize=9)
    ax.set_ylabel("t-SNE dimension 2", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.tick_params(labelsize=7.5, length=3, colors="#666666")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, lw=0.4, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)

    ax.text(0.985, 0.015, metrics, transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.6, alpha=0.9))

    if show_legend:
        handles = [Line2D([], [], marker=mk, color="none", markerfacecolor=cl,
                          markeredgecolor="none", markersize=6, label=gn)
                   for gn, cl, mk, _ in GROUPS]
        ax.legend(handles=handles, title="Organ system", fontsize=7.4,
                  title_fontsize=8, loc="upper left", frameon=True, framealpha=0.92,
                  edgecolor="#cccccc", borderpad=0.6, labelspacing=0.5, handletextpad=0.5)


def main():
    met = json.load(open(f"{OUT}/tsne_quant_metrics.json"))

    # ---------- (a) 전처리된 이진 증상 행렬 ----------
    base = pd.read_csv(BASE)
    meta = ["eng_name", "Kor_name", "Major", "diseases"]
    sym = base[[c for c in base.columns if c not in meta]].apply(
        pd.to_numeric, errors="coerce").fillna(0)
    sym = sym.loc[:, sym.isin([0, 1]).all()]
    bs = pd.concat([base["Kor_name"], sym], axis=1).groupby(
        "Kor_name", group_keys=False).apply(
        lambda g: g.sample(n=min(PER_CLASS, len(g)), random_state=SEED)).reset_index(drop=True)
    Xa = PCA(n_components=50, random_state=SEED).fit_transform(bs[sym.columns].values)
    Za = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30).fit_transform(Xa)
    ya = [KOR2ENG.get(k, k) for k in bs["Kor_name"]]
    print("panel a done", Za.shape)

    # ---------- (b) 생성 문장 Sentence-BERT 임베딩 ----------
    df = pd.read_csv(TOTAL)
    sp = df.groupby("labels", group_keys=False).apply(
        lambda g: g.sample(n=min(PER_CLASS, len(g)), random_state=SEED)).reset_index(drop=True)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(met["embed_model"])
    emb = st.encode(sp["질문"].astype(str).tolist(), batch_size=128,
                    convert_to_numpy=True, show_progress_bar=False)
    Zb = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30).fit_transform(emb)
    yb = [KOR2ENG.get(k, k) for k in sp["labels"]]
    print("panel b done", Zb.shape)

    ma = (f"silhouette = {met['processed_symptom_silhouette']:.2f}\n"
          f"kNN-10 label agreement = {met['processed_symptom_knn10_agreement']*100:.1f}%")
    mb = (f"silhouette = {met['generated_silhouette']:.2f}\n"
          f"kNN-10 label agreement = {met['generated_knn10_agreement']*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6))
    fig.patch.set_facecolor("white")
    panel(axes[0], Za, ya, "(a) Processed symptom data (binary symptom matrix, PCA-50)", ma, True)
    panel(axes[1], Zb, yb, "(b) SHAPE-generated sentences (Sentence-BERT embeddings)", mb, False)
    fig.tight_layout(w_pad=2.4)
    fig.savefig(f"{OUT}/Figure4.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/Figure4.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for tag, Z, y, m, t in (("a", Za, ya, ma, "(a) Processed symptom data (binary symptom matrix, PCA-50)"),
                            ("b", Zb, yb, mb, "(b) SHAPE-generated sentences (Sentence-BERT embeddings)")):
        f, a = plt.subplots(figsize=(8.4, 7.6))
        f.patch.set_facecolor("white")
        panel(a, Z, y, t, m, True)
        f.tight_layout()
        f.savefig(f"{OUT}/Figure4{tag}.png", dpi=400, bbox_inches="tight", facecolor="white")
        plt.close(f)

    print("saved:", f"{OUT}/Figure4.png / .pdf / Figure4a.png / Figure4b.png")


if __name__ == "__main__":
    main()
