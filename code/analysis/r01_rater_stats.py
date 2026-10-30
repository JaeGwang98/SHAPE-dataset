# -*- coding: utf-8 -*-
"""R7-2, R9-3: 전문가 3인 원점수 -> 평가자 매핑, 2축 분리, 점수 분포, 평가자 간 일치도."""
import pandas as pd, numpy as np, json, itertools
from sklearn.metrics import cohen_kappa_score
D = "../데이터 검증_390/"
F = {"0605": "첨부.생성데이터_검토_0605.xlsx", "0611": "첨부.생성데이터_평가요청_06.11.xlsx",
     "0611sev": "첨부.생성데이터_평가요청_06.11-raterC.xlsx"}
raw = pd.read_excel(D + F["0605"], header=None)
print("INSTRUCTIONS:\n", raw.iloc[2, 0])
R = {}
for k, f in F.items():
    d = pd.read_excel(D + f, header=None).iloc[6:396, :4]
    d.columns = ["no", "q", "label", "score"]
    R[k] = d.reset_index(drop=True)
base = R["0605"]
for k in R:
    assert (R[k]["q"] == base["q"]).all() and (R[k]["label"] == base["label"]).all(), k
S = pd.DataFrame({k: pd.to_numeric(R[k]["score"], errors="coerce") for k in R})
S["label"] = base["label"]
print("missing:", S[list(F)].isna().sum().to_dict(), "n:", len(S))
tab = pd.read_excel(D + "질병별_평균_평가점수_테이블.xlsx")
means = S.groupby("label")[list(F)].mean().round(2)
mt = tab.set_index("질병명")
for col in ["평균 점수_A", "평균 점수_B", "평균 점수_C"]:
    best = min(F, key=lambda k: (means[k] - mt[col].reindex(means.index)).abs().sum())
    print(col, "->", best, "abs diff sum", round((means[best] - mt[col].reindex(means.index)).abs().sum(), 3))

S = S.rename(columns={"0605": "A", "0611": "B", "0611sev": "C"}); E = ["A", "B", "C"]
out = {}
# 점수 분포
dist = {e: S[e].value_counts().reindex([1, 2, 3, 4], fill_value=0).astype(int).to_dict() for e in E}
allv = pd.concat([S[e] for e in E]); dist["pooled"] = allv.value_counts().reindex([1,2,3,4], fill_value=0).astype(int).to_dict()
out["distribution"] = dist
# 2축: label correct = {3,4}, realistic = {2,4}
lab = S[E].isin([3, 4]); nat = S[E].isin([2, 4])
out["label_correct_pct_by_rater"] = (lab.mean() * 100).round(1).to_dict()
out["realistic_pct_by_rater"] = (nat.mean() * 100).round(1).to_dict()
out["label_correct_pooled_pct"] = round(lab.values.mean() * 100, 1)
out["realistic_pooled_pct"] = round(nat.values.mean() * 100, 1)
out["label_majority_correct_items"] = int((lab.sum(1) >= 2).sum())
out["label_unanimous_correct_items"] = int((lab.sum(1) == 3).sum())
out["realistic_majority_items"] = int((nat.sum(1) >= 2).sum())
# 일치도
def fleiss(M, cats):
    n = M.shape[1]; cnt = np.stack([(M == c).sum(1) for c in cats], 1)
    P = ((cnt * (cnt - 1)).sum(1) / (n * (n - 1))).mean(); p = cnt.sum(0) / cnt.sum()
    return (P - (p ** 2).sum()) / (1 - (p ** 2).sum())
def kripp_ordinal(M):
    # Krippendorff alpha, ordinal metric, complete data
    vals = np.sort(np.unique(M)); n_c = np.array([(M == v).sum() for v in vals]); N = n_c.sum()
    idx = {v: i for i, v in enumerate(vals)}
    def d2(i, j):
        lo, hi = min(i, j), max(i, j)
        return (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    m = M.shape[1]; Do = 0
    for row in M:
        for a, b in itertools.permutations(range(m), 2):
            Do += d2(idx[row[a]], idx[row[b]]) / (m - 1)
    Do /= N
    De = sum(n_c[i] * n_c[j] * d2(i, j) for i in range(len(vals)) for j in range(len(vals))) / (N * (N - 1))
    return 1 - Do / De
def icc2_1(M):
    n, k = M.shape; gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum(); mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
M = S[E].values.astype(int)
rng = np.random.default_rng(42)
def boot(fn, B=2000):
    v = [fn(M[rng.integers(0, len(M), len(M))]) for _ in range(B)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
out["fleiss_kappa_4cat"] = round(fleiss(M, [1, 2, 3, 4]), 3)
out["kripp_alpha_ordinal"] = round(kripp_ordinal(M), 3); out["kripp_alpha_ordinal_CI"] = boot(kripp_ordinal, 1000)
out["icc2_1"] = round(icc2_1(M.astype(float)), 3); out["icc2_1_CI"] = boot(lambda X: icc2_1(X.astype(float)))
L = lab.values.astype(int); Nn = nat.values.astype(int)
out["fleiss_kappa_label_axis"] = round(fleiss(L, [0, 1]), 3)
out["fleiss_kappa_realism_axis"] = round(fleiss(Nn, [0, 1]), 3)
out["pairwise_weighted_kappa_quadratic"] = {f"{a}-{b}": round(cohen_kappa_score(S[a], S[b], weights="quadratic"), 3) for a, b in itertools.combinations(E, 2)}
out["pct_exact_agreement_all3"] = round(float((M.max(1) == M.min(1)).mean() * 100), 1)
out["pct_within1_all3"] = round(float((M.max(1) - M.min(1) <= 1).mean() * 100), 1)
out["prevalence_note"] = "high prevalence of score 4 depresses kappa (kappa paradox)"
# 질병별 2축
per = pd.DataFrame({"label_correct_pct": lab.groupby(S["label"]).apply(lambda g: g.values.mean() * 100).round(1),
                    "realistic_pct": nat.groupby(S["label"]).apply(lambda g: g.values.mean() * 100).round(1),
                    "mean": S.groupby("label")[E].mean().mean(1).round(2)})
per.to_csv("round2/rater_per_disease.csv", encoding="utf-8-sig")
json.dump(out, open("round2/rater_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)); print(per.sort_values("label_correct_pct").to_string())
