# -*- coding: utf-8 -*-
"""R7-1, R8-1, R9-3: RI 증강 130문항 의사 평가(평가자 A/B/C, 2026-10 회신) 집계 -> ri_review_results.json, ri_review_per_disease.csv.
입력: SHAPE_Revision/ 회신 시트 2개 (A = raterA, B·C = 평가자B&C). 문항·라벨 순서는 KEY 시트와 대조."""
import json, itertools, re, pandas as pd, numpy as np, openpyxl
from sklearn.metrics import cohen_kappa_score
IN = "../../../SHAPE_Revision/"
SRC = {"A": ("RI_expert_review_sheet_raterA.xlsx", "평가자A"),
       "B": ("RI_expert_review_sheet_평가자B&C.xlsx", "평가자B"), "C": ("RI_expert_review_sheet_평가자B&C.xlsx", "평가자C")}
key = pd.read_excel("RI_expert_review_sheet.xlsx", sheet_name="KEY_평가자에게_보내지_말것")
ref = pd.read_excel("RI_expert_review_sheet.xlsx", sheet_name="평가자A")
E = list(SRC); R = {}
for e, (f, s) in SRC.items():
    rows = list(openpyxl.load_workbook(IN + f, data_only=True)[s].iter_rows(values_only=True))
    d = pd.DataFrame(rows[1:], columns=["id", "q", "label", "real", "ok", "plaus", "alt", "note"]).dropna(subset=["id"]).reset_index(drop=True)
    assert len(d) == 130 and (d.q.values == ref["질문"].values).all() and (d.label.values == ref["질병명"].values).all(), e
    ok = d.ok.astype(str).str.strip()
    # 평가자 B 120번: ② 칸에 '질염'만 기재 -> 아니오(대안 질염)로 처리
    fix = ~ok.isin(["예", "아니오"])
    d.loc[fix, "alt"] = ok[fix]; ok[fix] = "아니오"
    d["ok"] = (ok == "예").astype(int); d["real"] = d.real.astype(float).astype(int); d["plaus"] = d.plaus.astype(float).astype(int)
    R[e] = d
    print(e, "ok-field fixes:", d.loc[fix, "id"].tolist())
assert (key["연번"].values == R["A"]["id"].astype(int).values).all()

def fleiss(M, cats):
    n = M.shape[1]; cnt = np.stack([(M == c).sum(1) for c in cats], 1)
    P = ((cnt * (cnt - 1)).sum(1) / (n * (n - 1))).mean(); p = cnt.sum(0) / cnt.sum()
    return (P - (p ** 2).sum()) / (1 - (p ** 2).sum())
def kripp(M, metric="ordinal"):
    vals = np.sort(np.unique(M)); n_c = np.array([(M == v).sum() for v in vals]); N = n_c.sum(); idx = {v: i for i, v in enumerate(vals)}
    def d2(i, j):
        if metric == "nominal": return float(i != j)
        lo, hi = min(i, j), max(i, j); return (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    m = M.shape[1]
    Do = sum(d2(idx[r[a]], idx[r[b]]) for r in M for a, b in itertools.permutations(range(m), 2)) / (m - 1) / N
    De = sum(n_c[i] * n_c[j] * d2(i, j) for i in range(len(vals)) for j in range(len(vals))) / (N * (N - 1))
    return 1 - Do / De
rng = np.random.default_rng(42)
def boot(M, fn, B=1000):
    v = [fn(M[rng.integers(0, len(M), len(M))]) for _ in range(B)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

real = np.column_stack([R[e].real for e in E]); plaus = np.column_stack([R[e].plaus for e in E]); ok = np.column_stack([R[e].ok for e in E])
r1 = lambda x: round(float(x), 1)
out = {"note": "2026-10-16, 평가자 A/B/C 회신 시트에서 계산 (r08_ri_review_stats.py). 비율·평균은 130문항 × 3인 = 390 평정 기준.",
       "n": 130, "n_ratings": 390,
       "label_ok_pct": r1(ok.mean() * 100), "realism_mean": round(float(real.mean()), 6), "realistic_pct": r1((real >= 3).mean() * 100),
       "plausibility_mean": round(float(plaus.mean()), 6), "plausible_pct": r1((plaus >= 3).mean() * 100),
       "label_majority_ok_items": int((ok.sum(1) >= 2).sum()), "label_unanimous_ok_items": int((ok.sum(1) == 3).sum()),
       "label_unanimous_not_ok_items": int((ok.sum(1) == 0).sum()),
       "plausible_majority_items": int(((plaus >= 3).sum(1) >= 2).sum()), "plaus_score1_ratings": int((plaus == 1).sum()),
       "by_rater": {e: {"label_ok_pct": r1(R[e].ok.mean() * 100), "realism_mean": round(float(R[e].real.mean()), 2),
                        "realism_dist": R[e].real.value_counts().reindex([1, 2, 3, 4], fill_value=0).astype(int).tolist(),
                        "plausibility_mean": round(float(R[e].plaus.mean()), 2),
                        "plaus_dist": R[e].plaus.value_counts().reindex([1, 2, 3, 4], fill_value=0).astype(int).tolist(),
                        "plausible_pct": r1((R[e].plaus >= 3).mean() * 100)} for e in E},
       "agreement": {"alpha_ordinal_realism": round(kripp(real), 3), "alpha_ordinal_realism_CI": boot(real, kripp),
                     "alpha_ordinal_plausibility": round(kripp(plaus), 3), "alpha_ordinal_plausibility_CI": boot(plaus, kripp),
                     "alpha_nominal_label": round(kripp(ok, "nominal"), 3), "alpha_nominal_label_CI": boot(ok, lambda X: kripp(X, "nominal")),
                     "fleiss_kappa_label": round(fleiss(ok, [0, 1]), 3),
                     "pct_all3_agree_label": r1((ok.min(1) == ok.max(1)).mean() * 100),
                     "weighted_kappa_plaus": {f"{a}-{b}": round(cohen_kappa_score(R[a].plaus, R[b].plaus, weights="quadratic"), 3) for a, b in itertools.combinations(E, 2)},
                     "kappa_label": {f"{a}-{b}": round(cohen_kappa_score(R[a].ok, R[b].ok), 3) for a, b in itertools.combinations(E, 2)}}}
out["alpha"] = out["agreement"]["alpha_ordinal_plausibility"]
# 원천 LLM별
for src in ["GPT-4o-mini", "GPT-4.1-mini"]:
    m = (key.source == src).values
    out.setdefault("by_source", {})[src] = {"n": int(m.sum()), "label_ok_pct": r1(ok[m].mean() * 100), "plausible_pct": r1((plaus[m] >= 3).mean() * 100),
                                             "realism_mean": round(float(real[m].mean()), 2)}
# 질병별
EN = dict(re.findall(r'"([^"]+)": "([^"]+)"', open("build_supplementary_r2.py", encoding="utf-8").read().split("EN = {")[1].split("}")[0]))
lab = R["A"].label.values
per = pd.DataFrame({"label": lab, "ok": ok.mean(1), "real": real.mean(1), "plaus": plaus.mean(1), "plaus_ok": (plaus >= 3).mean(1)}).groupby("label").agg(
    label_ok_pct=("ok", lambda x: round(x.mean() * 100, 1)), realism_mean=("real", "mean"), plausibility_mean=("plaus", "mean"), plausible_pct=("plaus_ok", lambda x: round(x.mean() * 100, 1)))
per["en"] = per.index.map(EN); per = per.sort_values("en")
per.to_csv("ri_review_per_disease.csv", encoding="utf-8-sig")
out["diseases_label_ok_ge70"] = int((per.label_ok_pct >= 70).sum())
out["lowest_label_ok"] = per.sort_values("label_ok_pct").head(4)[["en", "label_ok_pct"]].values.tolist()
json.dump(out, open("ri_review_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)); print(per.to_string())
