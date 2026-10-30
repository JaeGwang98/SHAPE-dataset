# -*- coding: utf-8 -*-
"""R9-6: KLUE-BERT b16, 78k (source ablation 3조건 x seed 45,46 = 6 모델) 항목별 예측 -> strict/any-match/macro-F1/balanced acc/혼동행렬/오류분석.
인자 없이 실행하면 기존과 동일(round2/pred_metrics.json, SuppFig_S1_confusion.png).
10-28 추가: 대표 모델(KLUE-BERT b32, 145,600, seed 42-46) 재계산용 인자
  python3 round2/r06_pred_metrics.py --glob "results/preds/R2_klue-bert_b32_s*.csv" --out_json round2/pred_metrics_b32.json \
      --out_fig round2/SuppFig_S1_confusion_b32.png --title "KLUE-BERT (batch 32)" --table round2/pred_metrics_b32_table \
      --orig_tag "A_klue-bert_b32_s{seed}"
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="results/preds/PM_pm_*_s4[56].csv")
ap.add_argument("--out_json", default="round2/pred_metrics.json")
ap.add_argument("--out_fig", default="round2/SuppFig_S1_confusion.png")
ap.add_argument("--title", default="KLUE-BERT (batch 16)")
ap.add_argument("--table", default=None, help="seed별 표(.csv/.md)·혼동행렬 csv 저장 경로 접두어")
ap.add_argument("--orig_tag", default=None, help="results/train_results.jsonl 의 원래 Table 4 태그 (예: A_klue-bert_b32_s{seed})")
args = ap.parse_args()
EN = {"외이염": "Otitis externa", "중이염": "Otitis media", "결막염": "Conjunctivitis", "안검염": "Blepharitis", "인후두염": "Laryngopharyngitis",
      "부비동염": "Sinusitis", "편도염": "Tonsillitis", "골관절염": "Osteoarthritis", "림프절염": "Lymphadenitis", "감기": "Common cold", "독감": "Influenza",
      "방광염": "Cystitis", "요도염": "Urethritis", "자궁경부염": "Cervicitis", "전립선염": "Prostatitis", "질염": "Vaginitis", "간염": "Hepatitis",
      "담낭염": "Cholecystitis", "위염": "Gastritis", "위장염": "Gastroenteritis", "구내염": "Stomatitis", "치주염": "Periodontitis",
      "접촉성 피부염": "Contact dermatitis", "지루성 피부염": "Seborrheic dermatitis", "기관지염": "Bronchitis", "폐렴": "Pneumonia"}
fs = sorted(glob.glob(args.glob))
if not fs: raise SystemExit("no preds yet")
rows = []; allp = []; allg = []; wrong = np.zeros(100)
for f in fs:
    d = pd.read_csv(f); gold = [[t.strip() for t in str(s).split(",") if t.strip()] for s in d["gold"]]
    prim = [x[0] for x in gold]; pred = d["pred"].tolist()
    am = np.array([p in x for p, x in zip(pred, gold)]); wrong += ~am
    eff = [p if p in x else x[0] for p, x in zip(pred, gold)]
    nl = np.array([len(x) for x in gold])
    rows.append({"seed": f, "anymatch": am.mean() * 100, "strict": np.mean(np.array(prim) == np.array(pred)) * 100,
                 "macro_f1": f1_score(eff, pred, average="macro", labels=sorted(set(eff)), zero_division=0) * 100,
                 "balanced": balanced_accuracy_score(eff, pred) * 100, **{f"am{k}": am[nl == k].mean() * 100 for k in [1, 2, 3]},
                 "n_f1_labels": len(set(eff))})
    allp += pred; allg += eff
R = pd.DataFrame(rows)
f = lambda c: f"{R[c].mean():.1f} ± {R[c].std(ddof=1):.1f}"
out = {"n_seeds": len(R), "anymatch_acc": f("anymatch"), "strict_primary_acc": f("strict"), "macro_f1": f("macro_f1"), "balanced_acc": f("balanced"),
       "anymatch_by_nlabels": {str(k): f(f"am{k}") for k in [1, 2, 3]}}
# 오류 분석: 5 seed 합산 오분류 쌍 (gold primary -> pred)
d0 = pd.read_csv(fs[0]); gold0 = [[t.strip() for t in str(s).split(",") if t.strip()] for s in d0["gold"]]
pairs = {}
for fp in fs:
    d = pd.read_csv(fp)
    for p, x in zip(d["pred"], gold0):
        if p not in x: pairs[(x[0], p)] = pairs.get((x[0], p), 0) + 1
top = sorted(pairs.items(), key=lambda kv: -kv[1])[:5]
n_err = sum(pairs.values())
out["top_confusions"] = [{"primary": EN[a], "pred": EN[b], "count": c} for (a, b), c in top]
out["error_summary_text"] = (f"Across the {len(fs)} models, the most frequent errors were " + "; ".join(f"{EN[a].lower()} predicted as {EN[b].lower()} ({c} of {n_err} errors)" for (a, b), c in top[:3]) + ".")
k = int((wrong == len(fs)).sum())
out["consistently_wrong_items"] = f"{k} questions were misclassified by all {len(fs)} models."
labs = sorted(set(allg) | set(allp), key=lambda x: EN[x])
cm = confusion_matrix(allg, allp, labels=labs)
fig, ax = plt.subplots(figsize=(11, 10)); im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
ax.set_xticklabels([EN[l] for l in labs], rotation=90, fontsize=9); ax.set_yticklabels([EN[l] for l in labs], fontsize=9)
for i in range(len(labs)):
    for j in range(len(labs)):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted label"); ax.set_ylabel("Reference label (accepted label if matched, otherwise primary label)")
ax.set_title(f"Confusion matrix, {args.title}, pooled over {len(fs)} models")
fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(args.out_fig, dpi=300)
json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
if args.table:
    # seed별 표 + 평균 ± SD + (선택) 원래 Table 4 any-match 비교 + TF-IDF char LR baseline
    R["seed"] = [int(re.search(r"_s(\d+)\.csv$", x).group(1)) for x in R["seed"]]
    cols = ["anymatch", "strict", "macro_f1", "balanced", "am1", "am2", "am3"]
    if args.orig_tag:
        orig = {}
        for l in open("results/train_results.jsonl", encoding="utf-8"):
            r = json.loads(l); orig[r["tag"]] = r.get("independent_accuracy")
        R["table4_anymatch"] = [orig[args.orig_tag.format(seed=s)] * 100 for s in R["seed"]]
        R["diff_vs_table4"] = R["anymatch"] - R["table4_anymatch"]
        cols = ["table4_anymatch", "anymatch", "diff_vs_table4"] + cols[1:]
    R = R[["seed"] + cols + ["n_f1_labels"]]
    R.to_csv(args.table + ".csv", index=False, encoding="utf-8-sig")
    T = json.load(open("round2/testset_baselines.json", encoding="utf-8"))["text_baselines_on_verification100"]["tfidf_char_LR"]
    hdr = {"table4_anymatch": "Table 4 any-match", "anymatch": "Any-match", "diff_vs_table4": "Δ vs Table 4", "strict": "Strict (primary)",
           "macro_f1": "Macro-F1", "balanced": "Balanced acc.", "am1": "Any-match, 1 label", "am2": "Any-match, 2 labels", "am3": "Any-match, 3 labels"}
    L = ["| Model | " + " | ".join(hdr[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for _, r in R.iterrows():
        L.append(f"| {args.title}, seed {int(r['seed'])} | " + " | ".join(f"{r[c]:+.1f}" if c == "diff_vs_table4" else f"{r[c]:.1f}" for c in cols) + " |")
    L.append(f"| **Mean ± SD (n = {len(R)})** | " + " | ".join(f"{R[c].mean():+.1f} ± {R[c].std(ddof=1):.1f}" if c == "diff_vs_table4" else f"{R[c].mean():.1f} ± {R[c].std(ddof=1):.1f}" for c in cols) + " |")
    tf = {"anymatch": T["anymatch_acc"], "strict": T["strict_primary_acc"], "macro_f1": T["macro_f1_anymatch_resolved"], "balanced": T["balanced_acc_anymatch_resolved"]}
    L.append("| TF-IDF (char) + LR baseline | " + " | ".join(f"{tf[c]:.1f}" if c in tf else "–" for c in cols) + " |")
    wi = [i for i in range(100) if wrong[i] == len(fs)]
    L += ["", f"- Top misclassified pairs (primary → predicted, pooled over {len(fs)} models, {n_err} errors): "
          + "; ".join(f"{EN[a]} → {EN[b]} ({c})" for (a, b), c in top),
          f"- Questions misclassified by all {len(fs)} models: {k} (idx {', '.join(map(str, wi))})",
          f"- Macro-F1 label count per seed: {', '.join(map(str, R['n_f1_labels']))} (labels = set of reference labels; periodontitis absent from test set)"]
    open(args.table + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    pd.DataFrame(cm, index=[EN[l] for l in labs], columns=[EN[l] for l in labs]).to_csv(args.table + "_confusion.csv", encoding="utf-8-sig")
    print("\n".join(L))
