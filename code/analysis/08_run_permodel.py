# -*- coding: utf-8 -*-
"""모델별 이질성 ablation 실행: 3 arm x 5 seed = 15 학습, 2 GPU 분산, resume 지원."""
import json, os, subprocess, threading, queue
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results/permodel_results.jsonl")
LOG = os.path.join(HERE, "results/logs"); os.makedirs(LOG, exist_ok=True)
ARMS = {"pm_4o": "results/pm_4o.csv", "pm_41": "results/pm_41.csv", "pm_mix": "results/pm_mix.csv"}
SEEDS = [42, 43, 44, 45, 46]
def done():
    s=set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try: s.add(json.loads(l)["tag"])
            except: pass
    return s
def jobs():
    j=[]
    for arm,csv in ARMS.items():
        for sd in SEEDS:
            j.append(dict(tag=f"PM_{arm}_s{sd}", args=[
                "--model","klue-bert","--batch","16","--seed",str(sd),
                "--epochs","3","--mode","ablation","--train_csv",csv]))
    return j
def worker(gpu,q):
    while True:
        try: job=q.get_nowait()
        except queue.Empty: return
        tag=job["tag"]
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false",
                 HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")
        cmd=["python3","03_train.py","--tag",tag,"--out",OUT]+job["args"]
        print(f"[gpu{gpu}] START {tag}",flush=True)
        with open(os.path.join(LOG,f"{tag}.log"),"w") as lf:
            rc=subprocess.run(cmd,cwd=HERE,env=env,stdout=lf,stderr=subprocess.STDOUT).returncode
        print(f"[gpu{gpu}] {'OK' if rc==0 else 'FAIL'+str(rc)} {tag}",flush=True)
        q.task_done()
def main():
    todo=[j for j in jobs() if j["tag"] not in done()]
    print(f"남은 작업 {len(todo)}/9",flush=True)
    q=queue.Queue()
    for j in todo: q.put(j)
    ts=[threading.Thread(target=worker,args=(g,q)) for g in [0,1]]
    for t in ts: t.start()
    for t in ts: t.join()
    print("=== permodel ablation 종료 ===",flush=True)
if __name__=="__main__": main()
