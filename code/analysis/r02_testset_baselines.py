# -*- coding: utf-8 -*-
"""R7-3, R9-1, R9-4, R9-5, R9-6, R8-4: test set 구성/중복, Kaggle 전처리 건수, 고전 ML baseline."""
import pandas as pd, numpy as np, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
R = ".."; U = f"{R}/2025_05_19_data/사용할 데이터"
out = {}
v = pd.read_csv(f"{U}/검증데이터/verification_100.csv")
gold = [[t.strip() for t in str(s).split(",") if t.strip()] for s in v["질병명_리스트"]]
prim = [g[0] for g in gold]
tr = pd.read_csv(f"{U}/25.05.19/Trainset.csv"); tot = pd.read_csv(f"{U}/25.06.04/total_dataset.csv")
fs = pd.concat([pd.read_csv(f"{R}/{f}") for f in ["4o_example.csv", "4o-mini_example.csv", "4.1-mini_example.csv"]])["result"].astype(str)
# --- test set composition
out["test_n"] = len(v)
out["labels_per_item"] = pd.Series([len(g) for g in gold]).value_counts().sort_index().to_dict()
out["primary_label_counts"] = pd.Series(prim).value_counts().to_dict()
out["classes_absent_from_test"] = sorted(set(tr["labels"]) - set(t for g in gold for t in g))
# --- overlap (exact + near-duplicate, char 2-4gram TF-IDF cosine)
S = set(tot["질문"].astype(str)) | set(fs)
out["exact_overlap_with_corpus_or_fewshot"] = int(v["질문"].isin(S).sum())
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit(tot["질문"].astype(str))
Xv = vec.transform(v["질문"].astype(str))
def maxsim(texts):
    m = np.zeros(len(v))
    for i in range(0, len(texts), 20000):
        m = np.maximum(m, cosine_similarity(Xv, vec.transform(texts[i:i+20000])).max(1))
    return m
ms_corpus = maxsim(tot["질문"].astype(str).tolist()); ms_fs = maxsim(fs.tolist())
# 기준: corpus 내부 동일 질병 무작위 쌍의 유사도 분포
rng = np.random.default_rng(42); idx = rng.choice(len(tot), 4000, replace=False)
Xs = vec.transform(tot["질문"].astype(str).iloc[idx]); C = cosine_similarity(Xs); np.fill_diagonal(C, 0)
nn_within_corpus = C.max(1)
out["nearest_neighbor_cosine_test_to_corpus"] = {"mean": round(float(ms_corpus.mean()), 3), "max": round(float(ms_corpus.max()), 3), "n_ge_0.8": int((ms_corpus >= .8).sum()), "n_ge_0.9": int((ms_corpus >= .9).sum())}
out["nearest_neighbor_cosine_test_to_fewshot"] = {"mean": round(float(ms_fs.mean()), 3), "max": round(float(ms_fs.max()), 3), "n_ge_0.8": int((ms_fs >= .8).sum())}
out["reference_nn_cosine_corpus_internal_4000sample"] = {"mean": round(float(nn_within_corpus.mean()), 3)}
# --- Kaggle preprocessing counts
k = pd.read_csv(f"{R}/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
vc = k["diseases"].value_counts()
out["kaggle"] = {"rows": len(k), "diseases": int(k["diseases"].nunique()), "symptom_cols": k.shape[1] - 1,
                 "diseases_ge100": int((vc >= 100).sum()), "rows_ge100": int(vc[vc >= 100].sum())}
p = pd.read_csv(f"{R}/Final_data_0513.csv")
out["processed"] = {"rows": len(p), "diseases": int(p["Kor_name"].nunique()),
                    "rows_per_disease_min_max": [int(p["Kor_name"].value_counts().min()), int(p["Kor_name"].value_counts().max())]}
symcols = [c for c in p.columns if c not in ["eng_name", "Kor_name", "Major", "diseases"]]
out["processed"]["symptoms_per_row_mean_sd"] = [round(float(p[symcols].sum(1).mean()), 2), round(float(p[symcols].sum(1).std()), 2)]
out["processed"]["rows_with_zero_symptoms"] = int((p[symcols].sum(1) == 0).sum())
out["processed"]["duplicate_symptom_rows_within_disease"] = int(p.duplicated(["Kor_name"] + symcols).sum())
# --- baselines
def score(pred):
    am = float(np.mean([pp in g for pp, g in zip(pred, gold)]))
    # 단일정답 변환: 예측이 허용 라벨이면 그 라벨, 아니면 1순위 라벨
    eff = [pp if pp in g else g[0] for pp, g in zip(pred, gold)]
    return {"anymatch_acc": round(am * 100, 1), "strict_primary_acc": round(accuracy_score(prim, pred) * 100, 1),
            "macro_f1_anymatch_resolved": round(f1_score(eff, pred, average="macro", labels=sorted(set(eff)), zero_division=0) * 100, 1),
            "balanced_acc_anymatch_resolved": round(balanced_accuracy_score(eff, pred) * 100, 1)}
B = {}
tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=300000, sublinear_tf=True)
Xtr = tv.fit_transform(tr["질문"].astype(str)); Xte = tv.transform(v["질문"].astype(str))
B["tfidf_char_LR"] = score(LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, tr["labels"]).predict(Xte))
B["tfidf_char_LinearSVM"] = score(LinearSVC(C=0.5).fit(Xtr, tr["labels"]).predict(Xte))
tw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True)
Xtr2 = tw.fit_transform(tr["질문"].astype(str)); Xte2 = tw.transform(v["질문"].astype(str))
B["tfidf_word_LR"] = score(LogisticRegression(max_iter=2000, C=4.0).fit(Xtr2, tr["labels"]).predict(Xte2))
out["text_baselines_on_verification100"] = B
# binary symptom features (different input modality: 5-fold stratified CV on processed rows)
X = p[symcols].values; y = p["Kor_name"].values
skf = StratifiedKFold(5, shuffle=True, random_state=42); Bb = {}
for name, clf in [("LR", LogisticRegression(max_iter=3000)), ("RF", RandomForestClassifier(300, random_state=42, n_jobs=-1)), ("BernoulliNB", BernoulliNB())]:
    pr = cross_val_predict(clf, X, y, cv=skf, n_jobs=1 if name == "RF" else 5)
    Bb[name] = {"acc": round(accuracy_score(y, pr) * 100, 1), "macro_f1": round(f1_score(y, pr, average="macro") * 100, 1),
                "balanced_acc": round(balanced_accuracy_score(y, pr) * 100, 1)}
out["binary_symptom_baselines_5foldCV_processed"] = Bb
json.dump(out, open("round2/testset_baselines.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
