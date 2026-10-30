# -*- coding: utf-8 -*-
"""
Reviewer 3-2 잔여 대응: intra-class diversity(클래스 내 다양성) 정량화.

리뷰어 3-2는 semantic consistency / inter-class separability / intra-class diversity /
symptom-disease consistency 네 가지를 요구했으나 기존 대응은 앞의 둘과 넷째만 다뤘다.
본 스크립트는 셋째를 채우고, 동시에 "클러스터가 실제 임상 의미가 아니라 프롬프트
템플릿을 반영한 것 아니냐"는 같은 코멘트의 지적에 대한 반증 근거를 산출한다.

지표
  - 임베딩: 클래스 내 평균 쌍별 코사인 거리 / 클래스 간 평균 거리 / 중심까지 평균 거리
  - 어휘  : distinct-1, distinct-2, type-token ratio, 어휘 크기
  - 길이  : 문장 길이 평균과 표준편차
표본 구성은 02_tsne_metrics.py와 동일(클래스당 300, seed 42, 동일 임베딩 모델).
산출물: results/intraclass_diversity.json, results/intraclass_diversity.csv
"""
import json, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_distances

DATA = "../2025_05_19_data/사용할 데이터"
TOTAL = f"{DATA}/25.06.04/total_dataset.csv"
BASE = "../Final_data_0513.csv"
PER_CLASS = 300
SEED = 42
OUT = "results"

# 원고 Table 4 표기 정본 (10_make_figure4.py의 KOR2ENG와 동일하게 유지할 것)
CANON_KOR2ENG = {
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


def ngram_stats(texts, n):
    """distinct-n: 고유 n-gram 수 / 전체 n-gram 수 (어절 단위)."""
    total, uniq = 0, set()
    for t in texts:
        toks = str(t).split()
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return (len(uniq) / total) if total else 0.0


def main():
    df = pd.read_csv(TOTAL)
    samp = (df.groupby("labels", group_keys=False)
              .apply(lambda g: g.sample(n=min(PER_CLASS, len(g)), random_state=SEED))
              .reset_index(drop=True))
    texts = samp["질문"].astype(str).tolist()
    ylab = samp["labels"].values
    print(f"표본 {len(samp)}건 / {samp['labels'].nunique()}개 클래스")

    from sentence_transformers import SentenceTransformer
    model_name = "jhgan/ko-sroberta-multitask"
    st = SentenceTransformer(model_name)
    emb = st.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    classes = sorted(pd.unique(ylab))
    kor2eng = {}
    try:
        base = pd.read_csv(BASE)
        kor2eng = (base.dropna(subset=["Kor_name", "eng_name"])
                       .drop_duplicates("Kor_name")
                       .set_index("Kor_name")["eng_name"].to_dict())
    except Exception:
        pass
    # 원자료 eng_name에 비표준 표기가 섞여 있어(기관지염→tracheitis 등)
    # 원고 Table 4 표기를 정본으로 덮어쓴다. 10_make_figure4.py의 KOR2ENG와 동일.
    kor2eng.update(CANON_KOR2ENG)

    rows = []
    centroids = {}
    for c in classes:
        m = ylab == c
        E, T = emb[m], [t for t, k in zip(texts, m) if k]
        D = cosine_distances(E)
        iu = np.triu_indices(len(E), k=1)
        cen = E.mean(axis=0)
        centroids[c] = cen
        toks = [len(str(t).split()) for t in T]
        vocab = set(w for t in T for w in str(t).split())
        rows.append({
            "class_kor": c,
            "class_eng": kor2eng.get(c, c),
            "n": int(m.sum()),
            "intra_cosine_distance_mean": float(D[iu].mean()),
            "intra_cosine_distance_sd": float(D[iu].std()),
            "dist_to_centroid_mean": float(cosine_distances(E, cen.reshape(1, -1)).mean()),
            "distinct_1": ngram_stats(T, 1),
            "distinct_2": ngram_stats(T, 2),
            "type_token_ratio": len(vocab) / max(1, sum(toks)),
            "vocab_size": len(vocab),
            "len_mean": float(np.mean(toks)),
            "len_sd": float(np.std(toks)),
        })

    per = pd.DataFrame(rows).sort_values("intra_cosine_distance_mean")
    per.to_csv(f"{OUT}/intraclass_diversity.csv", index=False, encoding="utf-8-sig")

    # 클래스 간 거리는 반드시 '점 간 쌍별' 거리로 계산한다.
    # 중심(centroid) 간 거리는 평균을 취하며 전역 평균 쪽으로 수축하므로
    # 점 간 intra 거리와 직접 비교하면 분리도가 과소평가된다.
    D_all = cosine_distances(emb)
    same = (ylab[:, None] == ylab[None, :])
    iu_all = np.triu_indices(len(ylab), k=1)
    intra_pairs = D_all[iu_all][same[iu_all]]
    inter_pairs = D_all[iu_all][~same[iu_all]]
    intra = float(intra_pairs.mean())
    inter = float(inter_pairs.mean())
    cohen_d = float((inter - intra) /
                    np.sqrt((inter_pairs.var() + intra_pairs.var()) / 2))

    summary = {
        "embed_model": model_name,
        "per_class": PER_CLASS,
        "n_total": int(len(samp)),
        "n_classes": len(classes),
        "intra_class_cosine_distance_mean": intra,
        "intra_class_cosine_distance_sd": float(intra_pairs.std()),
        "inter_class_cosine_distance_mean": inter,
        "inter_class_cosine_distance_sd": float(inter_pairs.std()),
        "separation_ratio_inter_over_intra": inter / intra if intra else None,
        "separation_cohen_d": cohen_d,
        "distinct_1_corpus": ngram_stats(texts, 1),
        "distinct_2_corpus": ngram_stats(texts, 2),
        "distinct_1_within_class_mean": float(per["distinct_1"].mean()),
        "distinct_2_within_class_mean": float(per["distinct_2"].mean()),
        "type_token_ratio_within_class_mean": float(per["type_token_ratio"].mean()),
        "sentence_length_mean": float(per["len_mean"].mean()),
        "duplicate_rate": float(1 - len(set(texts)) / len(texts)),
        "most_diverse_class": per.iloc[-1]["class_eng"],
        "least_diverse_class": per.iloc[0]["class_eng"],
    }
    json.dump(summary, open(f"{OUT}/intraclass_diversity.json", "w"),
              ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n--- 클래스별 (다양성 낮은 순 5개) ---")
    print(per[["class_eng", "intra_cosine_distance_mean", "distinct_2", "vocab_size"]]
          .head(5).to_string(index=False))
    print("\n--- 클래스별 (다양성 높은 순 5개) ---")
    print(per[["class_eng", "intra_cosine_distance_mean", "distinct_2", "vocab_size"]]
          .tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
