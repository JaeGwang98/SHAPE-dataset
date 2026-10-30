# -*- coding: utf-8 -*-
"""모델별 이질성 ablation 집계: arm별 mean±SD (독립/내부)."""
import json, numpy as np, pandas as pd
rows=[json.loads(l) for l in open("results/permodel_results.jsonl",encoding="utf-8") if l.strip()]
df=pd.DataFrame(rows)
df["arm"]=df["train_csv"].str.extract(r"pm_([a-z0-9]+)\.csv")
name={"4o":"GPT-4o-mini only (78k)","41":"GPT-4.1-mini only (78k)","mix":"Mixed 50:50 (78k)"}
g=df.groupby("arm").agg(indep_m=("independent_accuracy","mean"),indep_s=("independent_accuracy","std"),
                        int_m=("internal_accuracy","mean"),int_s=("internal_accuracy","std"),
                        n=("seed","count")).reset_index()
order={"4o":0,"41":1,"mix":2}
g=g.sort_values("arm",key=lambda c:c.map(order))
print("=== Per-model heterogeneity ablation (KLUE-BERT, batch 16, mean±SD over seeds) ===")
for _,r in g.iterrows():
    print(f"{name.get(r.arm,r.arm):28s} independent={100*r.indep_m:.1f} ± {100*(r.indep_s or 0):.1f}  "
          f"internal={100*r.int_m:.1f} ± {100*(r.int_s or 0):.1f}  (n={int(r.n)})")
g.to_csv("results/permodel_summary.csv",index=False,encoding="utf-8-sig")
print("\nsaved results/permodel_summary.csv")
