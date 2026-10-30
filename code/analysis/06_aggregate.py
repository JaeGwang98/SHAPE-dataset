# -*- coding: utf-8 -*-
"""
스윕 결과 집계 -> 리뷰어 대응용 표. (부분 결과도 처리)
산출물:
  results/table4_repro.csv        : model x batch, mean±SD (독립/내부 정확도)  [R3-4, R4-7, R1-3]
  results/ablation_summary.csv     : arm별 mean±SD                              [R4-4]
  results/cv_summary.csv           : fold별 + mean±SD                            [R1-3]
  콘솔에 요약 출력
"""
import json, os, glob
import numpy as np, pandas as pd

R = "results"

def load(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ---- A) Table 4 재현 ----
a = [r for r in load(f"{R}/train_results.jsonl") if r.get("mode") == "holdout"]
if a:
    df = pd.DataFrame(a)
    g = df.groupby(["model", "batch"]).agg(
        indep_mean=("independent_accuracy", "mean"),
        indep_std=("independent_accuracy", "std"),
        intern_mean=("internal_accuracy", "mean"),
        intern_std=("internal_accuracy", "std"),
        n=("seed", "count")).reset_index()
    g["independent (mean±SD)"] = g.apply(
        lambda r: f"{100*r.indep_mean:.1f} ± {100*(r.indep_std or 0):.1f}", axis=1)
    g["internal (mean±SD)"] = g.apply(
        lambda r: f"{100*r.intern_mean:.1f} ± {100*(r.intern_std or 0):.1f}", axis=1)
    piv = g.pivot(index="batch", columns="model", values="independent (mean±SD)")
    piv.to_csv(f"{R}/table4_repro.csv", encoding="utf-8-sig")
    print("=== Table 4 재현 (독립 검증셋 정확도 %, mean±SD over seeds) ===")
    print(piv.to_string())
    print(f"\n(반복 수 n={g['n'].min()}~{g['n'].max()} seeds/조건)")
    print("\n[R4-7] 배치 크기 간 변동이 seed SD 범위 내인지 확인 -> 표의 ± 폭과 비교")
else:
    print("Table4: 아직 결과 없음")

# ---- B) CV ----
cv = load(f"{R}/cv_results.jsonl")
if cv:
    r = cv[-1]
    print("\n=== 5-fold CV (klue-bert) ===")
    print(f"독립: {100*r['cv_independent_mean']:.1f} ± {100*r['cv_independent_std']:.1f}")
    print(f"내부: {100*r['cv_internal_mean']:.1f} ± {100*r['cv_internal_std']:.1f}")
    pd.DataFrame({"fold": range(len(r["folds_independent"])),
                  "independent": r["folds_independent"],
                  "internal": r["folds_internal"]}).to_csv(
        f"{R}/cv_summary.csv", index=False, encoding="utf-8-sig")
else:
    print("\nCV: 아직 결과 없음")

# ---- C) Ablation ----
ab = load(f"{R}/ablation_results.jsonl")
if ab:
    df = pd.DataFrame(ab)
    df["arm"] = df["train_csv"].str.extract(r"abl_([a-z_]+)\.csv")
    s = df.groupby("arm").agg(
        indep_mean=("independent_accuracy", "mean"),
        indep_std=("independent_accuracy", "std"),
        n_train=("n_train", "first"),
        n=("seed", "count")).reset_index()
    order = {"core": 0, "rs": 1, "rs_ri": 2, "orig_match": 3}
    s = s.sort_values("arm", key=lambda c: c.map(order))
    s["independent (mean±SD)"] = s.apply(
        lambda r: f"{100*r.indep_mean:.1f} ± {100*(r.indep_std or 0):.1f}", axis=1)
    s.to_csv(f"{R}/ablation_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== RS/RI Ablation (독립 검증셋 정확도 %) ===")
    print(s[["arm", "n_train", "independent (mean±SD)", "n"]].to_string(index=False))
    print("core→rs→rs_ri 증가 = 증강 효과 / rs_ri vs orig_match = 증강 vs 동일량 실제문장")
else:
    print("\nAblation: 아직 결과 없음")
