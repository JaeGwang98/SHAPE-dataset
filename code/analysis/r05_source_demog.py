# -*- coding: utf-8 -*-
"""R9-3/정확성: 전문가 평가 390개의 생성 모델 출처별 점수, few-shot pool 사용, R8-3: 생성문 성별/연령 기술 분포."""
import json, re, pandas as pd, numpy as np
D = "../2025_05_19_data"; U = f"{D}/사용할 데이터"
val = pd.read_csv(f"{U}/검증데이터/Val_390.csv")
src = {}
for f, s in [("4o_origin_5200.csv", "GPT-4o"), ("4o_mini_origin_5200.csv", "GPT-4o-mini"), ("4.1_mini_origin_5200.csv", "GPT-4.1-mini")]:
    for q in pd.read_csv(f"{U}/{f}")["질문"]: src[q] = s
val["source"] = val["질문"].map(src)
E = ["../데이터 검증_390/첨부.생성데이터_검토_0605.xlsx", "../데이터 검증_390/첨부.생성데이터_평가요청_06.11.xlsx", "../데이터 검증_390/첨부.생성데이터_평가요청_06.11-raterC.xlsx"]
Sc = pd.DataFrame({k: pd.to_numeric(pd.read_excel(f, header=None).iloc[6:396, 3]).values for k, f in zip("ABC", E)})
q0 = pd.read_excel(E[0], header=None).iloc[6:396, 1].astype(str).values
assert (q0 == val["질문"].astype(str).values).all()
Sc["source"] = val["source"].values
out = {"val390_by_source": val["source"].value_counts().to_dict()}
g = Sc.groupby("source")
out["mean_score_by_source"] = g[list("ABC")].mean().mean(1).round(2).to_dict()
out["label_correct_pct_by_source"] = g.apply(lambda x: round(float(x[list("ABC")].isin([3, 4]).values.mean() * 100), 1)).to_dict()
out["realistic_pct_by_source"] = g.apply(lambda x: round(float(x[list("ABC")].isin([2, 4]).values.mean() * 100), 1)).to_dict()
corp = Sc[Sc.source != "GPT-4o"]
out["corpus_only_248"] = {"n": len(corp), "mean": round(float(corp[list("ABC")].values.mean()), 2),
    "label_correct_pct": round(float(corp[list("ABC")].isin([3, 4]).values.mean() * 100), 1),
    "realistic_pct": round(float(corp[list("ABC")].isin([2, 4]).values.mean() * 100), 1)}
lab = pd.read_excel(E[0], header=None).iloc[6:396, 2].values
cm = Sc.assign(label=lab)[Sc.source != "GPT-4o"].groupby("label")[list("ABC")].mean().mean(1)
out["corpus_only_diseases_mean_ge3"] = int((cm >= 3).sum())
# few-shot pool: 각 배치의 system prompt 예시가 어느 pool에서 왔는지
pools = {n: set(pd.read_csv(f"../{n}_example.csv")["result"].astype(str).str.strip()) for n in ["4o", "4o-mini", "4.1-mini"]}
for b in ["batch_4o-mini_50000_random_fewshot.jsonl", "batch_4.1-mini_50000_random_fewshot.jsonl", "batch_4o_origin_random_fewshot.jsonl"]:
    sp = json.loads(open(f"{D}/{b}", encoding="utf-8").readline())["body"]["messages"][0]["content"]
    ex = [l.strip() for l in sp.split("참고하여 질문 1개를 구성하세요.\n")[1].split("\n") if l.strip()]
    out[f"fewshot_pool_{b}"] = {n: sum(e in p for e in ex) for n, p in pools.items()}
# 성별/연령 (R8-3), 전체 156k
tot = pd.read_csv(f"{U}/25.06.04/total_dataset.csv")
t = tot["질문"].astype(str)
male = t.str.contains(r"남성|남자|남편|아들|할아버지|아저씨"); female = t.str.contains(r"여성|여자|아내|딸|할머니|임신|생리")
age = t.str.extract(r"(\d{1,2})\s*(?:세|살)")[0].astype(float)
dec = t.str.extract(r"(\d)0\s*대")[0].astype(float) * 10 + 5
agev = age.fillna(dec)
tot = tot.assign(male=male & ~female, female=female & ~male, age=agev)
out["demog_overall"] = {"pct_male_only": round(tot.male.mean() * 100, 1), "pct_female_only": round(tot.female.mean() * 100, 1),
                        "pct_age_stated": round(tot.age.notna().mean() * 100, 1), "age_median": float(tot.age.median()),
                        "age_IQR": [float(tot.age.quantile(.25)), float(tot.age.quantile(.75))],
                        "pct_age_ge65_of_stated": round(float((tot.age >= 65).mean() / tot.age.notna().mean() * 100), 1),
                        "pct_age_lt20_of_stated": round(float((tot.age < 20).mean() / tot.age.notna().mean() * 100), 1)}
per = tot.groupby("labels").agg(male=("male", "mean"), female=("female", "mean"), age_med=("age", "median")).round(3)
per["male_share_of_sexed"] = (per.male / (per.male + per.female)).round(3)
per.to_csv("round2/demographics_per_disease.csv", encoding="utf-8-sig")
out["demog_sex_specific"] = per.loc[["전립선염", "자궁경부염", "질염", "요도염", "방광염", "골관절염"]].to_dict("index")
out["demog_male_share_range"] = [float(per.male_share_of_sexed.min()), float(per.male_share_of_sexed.max())]
json.dump(out, open("round2/source_demog.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
print(per.sort_values("male_share_of_sexed").to_string())
