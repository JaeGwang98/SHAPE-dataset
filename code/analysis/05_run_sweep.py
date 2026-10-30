# -*- coding: utf-8 -*-
"""
실험 스케줄러: 2개 GPU에 작업을 분산, 각 GPU당 1개씩 순차 실행.
- 이미 완료된 tag(train_results.jsonl 내)은 건너뜀(resume).
- 로그: results/logs/<tag>.log
작업 구성:
  A) Table 4 재현 + 통계 : 3 model x 3 batch x 5 seed = 45  (mode=holdout)
  B) 5-fold CV           : klue-bert 1 x seed 5개 = 5        (mode=cv)
  C) RS/RI ablation      : klue-bert x 4 arm x 5 seed = 20   (mode=ablation)
"""
import json, os, subprocess, threading, queue

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "results/train_results.jsonl")
CVOUT  = os.path.join(HERE, "results/cv_results.jsonl")
ABLOUT = os.path.join(HERE, "results/ablation_results.jsonl")
LOGDIR = os.path.join(HERE, "results/logs")
os.makedirs(LOGDIR, exist_ok=True)
GPUS = [0, 1]
N_PER_GPU = 2   # GPU당 동시 실행 작업 수 (24GB 중 여유 충분: 모델당 ~2.5GB)

MODELS = ["klue-bert", "roberta", "deberta"]
BATCHES = [32, 16, 8]   # 빠른(batch 큰) 작업부터 -> 부분 결과/조기 검증 우선
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 3

def done_tags(path):
    tags = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: tags.add(json.loads(line)["tag"])
            except Exception: pass
    return tags

def build_jobs():
    jobs = []
    # A) Table4 재현 + 반복
    for m in MODELS:
        for b in BATCHES:
            for s in SEEDS:
                tag = f"A_{m}_b{b}_s{s}"
                jobs.append(dict(tag=tag, out=RESULT, args=[
                    "--model", m, "--batch", str(b), "--seed", str(s),
                    "--epochs", str(EPOCHS), "--mode", "holdout"]))
    # B) 5-fold CV (best model, 1회면 fold별 mean±SD 보고에 충분)
    jobs.append(dict(tag="B_cv_klue_s42", out=CVOUT, args=[
        "--model", "klue-bert", "--batch", "16", "--seed", "42",
        "--epochs", str(EPOCHS), "--mode", "cv", "--folds", "5"]))
    # C) ablation
    arms = {"core": "results/abl_core.csv", "rs": "results/abl_rs.csv",
            "rs_ri": "results/abl_rs_ri.csv", "orig_match": "results/abl_orig_match.csv"}
    for arm, csv in arms.items():
        for s in SEEDS:
            tag = f"C_abl_{arm}_s{s}"
            jobs.append(dict(tag=tag, out=ABLOUT, args=[
                "--model", "klue-bert", "--batch", "16", "--seed", str(s),
                "--epochs", str(EPOCHS), "--mode", "ablation", "--train_csv", csv]))
    return jobs

def worker(gpu, q):
    while True:
        try: job = q.get_nowait()
        except queue.Empty: return
        tag = job["tag"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONUNBUFFERED="1")
        # DeBERTa-v3는 transformers 5.4에서 학습 붕괴 -> 구버전 venv 파이썬 사용
        py = "/tmp/deberta_env/bin/python" if "deberta" in job["args"] else "python3"
        cmd = [py, "03_train.py", "--tag", tag, "--out", job["out"]] + job["args"]
        log = os.path.join(LOGDIR, f"{tag}.log")
        print(f"[gpu{gpu}] START {tag}", flush=True)
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        print(f"[gpu{gpu}] {'OK' if rc==0 else 'FAIL('+str(rc)+')'} {tag}", flush=True)
        q.task_done()

def main():
    all_jobs = build_jobs()
    done = done_tags(RESULT) | done_tags(CVOUT) | done_tags(ABLOUT)
    todo = [j for j in all_jobs if j["tag"] not in done]
    print(f"전체 {len(all_jobs)}개 중 완료 {len(done)}개, 남은 작업 {len(todo)}개", flush=True)
    q = queue.Queue()
    for j in todo: q.put(j)
    slots = [g for g in GPUS for _ in range(N_PER_GPU)]  # GPU당 N_PER_GPU 워커
    print(f"동시 실행 워커: {len(slots)}개 (GPU {GPUS} x {N_PER_GPU})", flush=True)
    threads = [threading.Thread(target=worker, args=(g, q)) for g in slots]
    for t in threads: t.start()
    for t in threads: t.join()
    print("=== 스윕 종료 ===", flush=True)

if __name__ == "__main__":
    main()
