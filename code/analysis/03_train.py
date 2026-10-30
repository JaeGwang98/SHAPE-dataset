# -*- coding: utf-8 -*-
"""
BERT 재학습 하니스 — 리뷰어 대응 실험 공용.
지원 모드:
  holdout : Trainset으로 학습 -> Testset(내부, 단일라벨) + verification_100(독립, 다중라벨 any-match) 평가
  cv      : total_dataset 156k에 대한 stratified K-fold 교차검증
  ablation: --train_csv 로 임의 학습셋 지정(무증강/증강 비교용)

평가 프로토콜(논문 Table 4 재현):
  - 독립 정확도 = verification_100 상위1 예측이 전문가 라벨(최대3개) 집합에 포함되면 정답 (any-match)
사용 예:
  python3 03_train.py --model klue-bert --batch 16 --seed 42 --mode holdout
"""
import argparse, json, os, random, sys
import numpy as np, pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

DATA = "../2025_05_19_data/사용할 데이터"
TRAIN_CSV = f"{DATA}/25.05.19/Trainset.csv"
TEST_CSV  = f"{DATA}/25.05.19/Testset.csv"
TOTAL_CSV = f"{DATA}/25.06.04/total_dataset.csv"
VERIF_CSV = f"{DATA}/검증데이터/verification_100.csv"

MODELS = {
    "klue-bert": "klue/bert-base",
    "roberta":   "klue/roberta-base",
    "deberta":   "microsoft/mdeberta-v3-base",
}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def build_labels():
    labs = sorted(pd.read_csv(TRAIN_CSV)["labels"].unique().tolist())
    return {l: i for i, l in enumerate(labs)}, {i: l for i, l in enumerate(labs)}

def tokenize_df(df, tok, l2i, max_len):
    ds = Dataset.from_dict({"text": df["질문"].astype(str).tolist(),
                            "label": [l2i[x] for x in df["labels"]]})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                  batched=True, remove_columns=["text"])

def independent_accuracy(model, tok, l2i, i2l, max_len, device, dump_path=None):
    """verification_100: top-1 예측이 전문가 라벨집합에 있으면 정답(any-match)."""
    v = pd.read_csv(VERIF_CSV)
    gold = [set(t.strip() for t in str(s).split(",")) for s in v["질병명_리스트"]]
    model.eval(); correct = 0; preds = []; probs = []
    with torch.no_grad():
        for i in range(0, len(v), 64):
            texts = v["질문"].astype(str).tolist()[i:i+64]
            enc = tok(texts, truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            probs.append(torch.softmax(logits, -1).cpu().numpy())
            pred = logits.argmax(-1).cpu().numpy()
            for j, p in enumerate(pred):
                preds.append(i2l[int(p)])
                if i2l[int(p)] in gold[i+j]:
                    correct += 1
    if dump_path:
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        # 26개 클래스 softmax 확률(prob_<라벨>)도 함께 저장 (10-28 추가)
        P = pd.DataFrame(np.concatenate(probs), columns=[f"prob_{i2l[k]}" for k in range(len(i2l))])
        pd.concat([pd.DataFrame({"idx": range(len(v)), "gold": v["질병명_리스트"], "pred": preds}), P], axis=1).to_csv(
            dump_path, index=False, encoding="utf-8-sig")
    return correct / len(v)

def train_one(model_name, train_df, eval_df, l2i, i2l, args, tag):
    tok = AutoTokenizer.from_pretrained(model_name)
    tr = tokenize_df(train_df, tok, l2i, args.max_len)
    ev = tokenize_df(eval_df, tok, l2i, args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(l2i))
    coll = DataCollatorWithPadding(tok)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": float((preds == p.label_ids).mean())}

    # DeBERTa-v2/v3: mixed precision(fp16/bf16)에서 초기 loss 폭발 후 ln(26)에 붕괴하는
    #   알려진 문제 -> fp32(전정밀도) + warmup + lr 하향으로 안정 학습. klue/roberta는 fp16 유지.
    is_deberta = "deberta" in model_name.lower()
    lr = args.lr  # deberta도 2e-5 사용하되 warmup으로 초기 폭발 방지
    targs = TrainingArguments(
        output_dir=f"/tmp/shape_ckpt/{tag}",
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=128,
        learning_rate=lr, num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio, max_grad_norm=1.0,
        eval_strategy="epoch", save_strategy="no", logging_steps=500,
        seed=args.seed, fp16=not is_deberta, bf16=is_deberta,
        report_to=[], dataloader_num_workers=4, disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=ev,
                      processing_class=tok, data_collator=coll, compute_metrics=metrics)
    trainer.train()
    internal = trainer.evaluate()["eval_accuracy"]
    device = model.device
    dump = f"results/preds/{tag}.csv" if args.dump_preds else None
    indep = independent_accuracy(model, tok, l2i, i2l, args.max_len, device, dump)
    del model, trainer; torch.cuda.empty_cache()
    return internal, indep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    # 07-13 Table 4 KLUE/RoBERTa·CV 실행은 로그상 warm-up 없음(첫 LR 1.927e-5 등) -> 재현용 0 지정 가능 (10-28)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--mode", choices=["holdout", "cv", "ablation"], default="holdout")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train_csv", default=None, help="ablation 모드: 학습셋 경로")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0, help=">0 이면 학습셋 앞 N개만 사용(스모크 테스트)")
    ap.add_argument("--out", default="results/train_results.jsonl")
    ap.add_argument("--dump_preds", action="store_true", help="verification_100 항목별 예측 저장(results/preds/)")
    args = ap.parse_args()

    set_seed(args.seed)
    l2i, i2l = build_labels()
    model_name = MODELS[args.model]
    tag = args.tag or f"{args.model}_b{args.batch}_s{args.seed}_{args.mode}"
    rec = {"model": args.model, "checkpoint": model_name, "batch": args.batch,
           "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
           "max_len": args.max_len, "warmup_ratio": args.warmup_ratio, "mode": args.mode, "tag": tag}

    if args.mode == "cv":
        tot = pd.read_csv(TOTAL_CSV)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_int, fold_ind = [], []
        for k, (tri, tei) in enumerate(skf.split(tot["질문"], tot["labels"])):
            it, ind = train_one(model_name, tot.iloc[tri], tot.iloc[tei],
                                l2i, i2l, args, f"{tag}_f{k}")
            fold_int.append(it); fold_ind.append(ind)
            print(f"[fold {k}] internal={it:.4f} independent={ind:.4f}", flush=True)
        # 표본표준편차(ddof=1) — 논문 Table 5/7/8 및 95% CI 계산과 동일 기준
        rec.update({"cv_internal_mean": float(np.mean(fold_int)),
                    "cv_internal_std": float(np.std(fold_int, ddof=1)),
                    "cv_independent_mean": float(np.mean(fold_ind)),
                    "cv_independent_std": float(np.std(fold_ind, ddof=1)),
                    "folds_internal": fold_int, "folds_independent": fold_ind})
    else:
        train_df = pd.read_csv(args.train_csv) if args.train_csv else pd.read_csv(TRAIN_CSV)
        if args.limit:
            train_df = train_df.groupby("labels", group_keys=False).head(max(1, args.limit // 26))
        eval_df = pd.read_csv(TEST_CSV)
        it, ind = train_one(model_name, train_df, eval_df, l2i, i2l, args, tag)
        rec.update({"internal_accuracy": it, "independent_accuracy": ind,
                    "train_csv": args.train_csv or TRAIN_CSV, "n_train": len(train_df)})
        print(f"internal={it:.4f} independent={ind:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("saved ->", args.out)

if __name__ == "__main__":
    main()
