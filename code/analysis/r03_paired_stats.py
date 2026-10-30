# -*- coding: utf-8 -*-
"""R9-2: seed 공유 -> 반복측정(paired) 통계, 효과크기, 95% CI. 존재하는 결과만 집계."""
import json, os, itertools
import numpy as np, pandas as pd
from scipy import stats

def multipletests(pv, method="holm"):
    pv = np.asarray(pv, float); o = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * pv[i])); adj[i] = run
    return None, adj
def anova_rm(w):
    # w: subjects(seed) x conditions
    Y = w.values.astype(float); n, k = Y.shape; gm = Y.mean()
    ss_c = n * ((Y.mean(0) - gm) ** 2).sum(); ss_s = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_e = ((Y - gm) ** 2).sum() - ss_c - ss_s; d1, d2 = k - 1, (k - 1) * (n - 1)
    F = (ss_c / d1) / (ss_e / d2); p = 1 - stats.f.cdf(F, d1, d2)
    return F, d1, d2, p, ss_c / (ss_c + ss_e)
RES = "results"
def load(f):
    p = os.path.join(RES, f)
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) if os.path.exists(p) else pd.DataFrame()
def ci_mean(x):
    x = np.asarray(x, float); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return [round(x.mean() - h, 6), round(x.mean() + h, 6)]
def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); t, p = stats.ttest_rel(a, b)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
    return {"mean_diff_pp": round(d.mean(), 6), "diff_95CI": ci_mean(d), "t": round(float(t), 3), "p": round(float(p), 4), "cohen_dz": round(float(dz), 6)}
def rm(df, cond):
    w = df.pivot(index="seed", columns=cond, values="acc").dropna()
    F, d1, d2, p, eta_p = anova_rm(w)
    fr = stats.friedmanchisquare(*[w[c] for c in w.columns]) if w.shape[1] >= 3 else None
    pairs = {}
    raw = []
    for x, y in itertools.combinations(w.columns, 2):
        pairs[f"{x} vs {y}"] = paired(w[x], w[y]); raw.append(pairs[f"{x} vs {y}"]["p"])
    if raw:
        adj = multipletests(raw, method="holm")[1]
        for k, pa in zip(pairs, adj): pairs[k]["p_holm"] = round(float(pa), 4)
    return {"n_seeds": int(len(w)), "means": {c: [round(w[c].mean(), 1), round(w[c].std(ddof=1), 1), ci_mean(w[c])] for c in w.columns},
            "rmANOVA": {"F": round(float(F), 6), "df": [int(d1), int(d2)], "p": round(float(p), 4), "partial_eta2": round(float(eta_p), 6)},
            "friedman": None if fr is None else {"chi2": round(float(fr.statistic), 3), "p": round(float(fr.pvalue), 4)}, "pairwise": pairs}
out = {}
A = load("train_results.jsonl")
if len(A):
    A["acc"] = A["independent_accuracy"] * 100
    out["batch_size"] = {m: rm(A[A.model == m], "batch") for m in A.model.unique()}
C = load("ablation_results.jsonl")
if len(C):
    C["acc"] = C["independent_accuracy"] * 100; C["arm"] = C["tag"].str.extract(r"C_abl_(.+)_s\d+")[0]
    out["sentence_level_EDA_ablation"] = rm(C, "arm")
P = load("permodel_results.jsonl")
if len(P):
    P["acc"] = P["independent_accuracy"] * 100; P["arm"] = P["tag"].str.extract(r"PM_pm_(.+)_s\d+")[0]
    out["source_ablation"] = rm(P, "arm")
RI = load("ri_ablation_results.jsonl")
if len(RI) >= 6:
    RI["acc"] = RI["independent_accuracy"] * 100; RI["arm"] = RI["tag"].str.extract(r"RI_ri_(.+)_s\d+")[0]
    out["shape_RI_ablation"] = rm(RI, "arm")
# equivalence (TOST, +-3pp margin) for key null comparisons
def tost(a, b, m=3.0):
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((d.mean() + m) / se, n - 1); p2 = stats.t.cdf((d.mean() - m) / se, n - 1)
    return {"margin_pp": m, "p_tost": round(float(max(p1, p2)), 4), "90CI": [round(d.mean() - stats.t.ppf(.95, n-1) * se, 6), round(d.mean() + stats.t.ppf(.95, n-1) * se, 6)]}
if "source_ablation" in out:
    w = P.pivot(index="seed", columns="arm", values="acc").dropna()
    out["source_ablation"]["TOST_mix_vs_4o"] = tost(w["mix"], w["4o"]); out["source_ablation"]["TOST_mix_vs_41"] = tost(w["mix"], w["41"])
if "shape_RI_ablation" in out:
    w = RI.pivot(index="seed", columns="arm", values="acc").dropna()
    if {"rsri", "rsonly"} <= set(w.columns): out["shape_RI_ablation"]["TOST_rsri_vs_rsonly"] = tost(w["rsri"], w["rsonly"])
json.dump(out, open("round2/paired_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

# 최소 검출 차이(MDE, 80% power, 양측 alpha .05): 5-seed paired 설계, 관측된 paired 차이 SD 중앙값 사용
sds = []
for key in ["sentence_level_EDA_ablation", "source_ablation", "shape_RI_ablation"]:
    if key in out:
        for v in out[key]["pairwise"].values():
            lo, hi = v["diff_95CI"]; n = out[key]["n_seeds"]
            sds.append((hi - lo) / 2 / stats.t.ppf(.975, n - 1) * np.sqrt(n))
if sds:
    n = 5; sd = float(np.median(sds))
    out["mde_pp"] = round((stats.t.ppf(.975, n - 1) + stats.t.ppf(.80, n - 1)) * sd / np.sqrt(n), 1)
    out["mde_sd_used"] = round(sd, 2)
json.dump(out, open("round2/paired_stats.json", "w"), ensure_ascii=False, indent=1)
print("MDE", out.get("mde_pp"), out.get("mde_sd_used"))
