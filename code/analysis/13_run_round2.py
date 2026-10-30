# -*- coding: utf-8 -*-
"""2차 리뷰 대응 GPU 스윕 (2 GPU, resume 지원).
  PM : source ablation seed 45,46 추가 (R9-2: 3-seed -> 5-seed)
  R2 : KLUE-BERT b32 x 5 seed, 항목별 예측 저장 (R9-6 macro-F1/confusion)
  RI : SHAPE RI ablation (RS-only / RS+RI 1:4 / RI-only, 크기 일치) x 5 seed (R6-2, R8-5)
"""
import json, os, subprocess, threading, queue
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "results/logs"); os.makedirs(LOG, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
def jobs():
    j = []
    for arm, csv in {"pm_4o": "results/pm_4o.csv", "pm_41": "results/pm_41.csv", "pm_mix": "results/pm_mix.csv"}.items():
        for sd in [45, 46]:
            j.append(dict(tag=f"PM_{arm}_s{sd}", out="results/permodel_results.jsonl", args=[
                "--model", "klue-bert", "--batch", "16", "--seed", str(sd), "--epochs", "3",
                "--mode", "ablation", "--train_csv", csv, "--dump_preds"]))
    for arm in ["ri_rsonly", "ri_rsri", "ri_rionly"]:
        for sd in SEEDS:
            j.append(dict(tag=f"RI_{arm}_s{sd}", out="results/ri_ablation_results.jsonl", args=[
                "--model", "klue-bert", "--batch", "16", "--seed", str(sd), "--epochs", "3",
                "--mode", "ablation", "--train_csv", f"results/{arm}.csv", "--dump_preds"]))
    for sd in SEEDS:
        j.append(dict(tag=f"R2_klue-bert_b32_s{sd}", out="results/round2_preds_results.jsonl", args=[
            "--model", "klue-bert", "--batch", "32", "--seed", str(sd), "--epochs", "3",
            "--mode", "holdout", "--dump_preds"]))
    # 10-28: Table 4 원래 실행과 같은 조건(warm-up 없음) 재학습
    for sd in SEEDS:
        j.append(dict(tag=f"R2nw_klue-bert_b32_s{sd}", out="results/round2_preds_nowarmup_results.jsonl", args=[
            "--model", "klue-bert", "--batch", "32", "--seed", str(sd), "--epochs", "3",
            "--mode", "holdout", "--warmup_ratio", "0", "--dump_preds"]))
    return j
def done():
    s = set()
    for f in ["permodel_results.jsonl", "ri_ablation_results.jsonl", "round2_preds_results.jsonl", "round2_preds_nowarmup_results.jsonl"]:
        p = os.path.join(HERE, "results", f)
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                try: s.add(json.loads(l)["tag"])
                except Exception: pass
    return s
def worker(gpu, q):
    while True:
        try: job = q.get_nowait()
        except queue.Empty: return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false",
                   HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")
        cmd = ["python3", "03_train.py", "--tag", job["tag"], "--out", job["out"]] + job["args"]
        print(f"[gpu{gpu}] START {job['tag']}", flush=True)
        with open(os.path.join(LOG, f"{job['tag']}.log"), "w") as lf:
            rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f"[gpu{gpu}] {'OK' if rc == 0 else 'FAIL' + str(rc)} {job['tag']}", flush=True)
def main():
    d = done(); todo = [j for j in jobs() if j["tag"] not in d]
    print(f"remaining {len(todo)}/{len(jobs())}", flush=True)
    q = queue.Queue()
    for j in todo: q.put(j)
    # GPUS 환경변수로 워커 지정 (예: GPUS=0,0 -> GPU0에 2개 동시). 기본 0,1
    gpus = [int(g) for g in os.environ.get("GPUS", "0,1").split(",")]
    ts = [threading.Thread(target=worker, args=(g, q)) for g in gpus]
    for t in ts: t.start()
    for t in ts: t.join()
    print("=== round2 sweep done ===", flush=True)
if __name__ == "__main__": main()
