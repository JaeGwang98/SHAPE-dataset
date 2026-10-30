# SHAPE 리뷰어 대응 실험 (API 불필요 버전)

모든 실험은 **기존 데이터 + 로컬 GPU**만 사용. OpenAI API 호출 없음.

## 실행 환경
- GPU: RTX 3090 ×2, torch 2.10+cu128, transformers 5.4, sentence-transformers 5.4
- 데이터: `../2025_05_19_data/사용할 데이터/` (Trainset 145,600 / Testset 15,600 / total 156,000 / verification_100 다중라벨)
- 모델: klue/bert-base, klue/roberta-base, lighthouse/mdeberta-v3-base-kor-further

## 평가 프로토콜 (논문 Table 4 재현)
- **독립 정확도**: `verification_100.csv`(전문가 최대 3개 라벨)에 대해 top-1 예측이 라벨집합에 포함되면 정답(any-match). 논문의 86%와 직접 비교 가능.
- **내부 정확도**: `Testset.csv`(단일 라벨, 동일 파이프라인) 정확도.

## 스크립트 ↔ 리뷰어 지적 매핑

| 스크립트 | 대응 리뷰어 지적 | 산출물 | API |
|---|---|---|---|
| `01_class_similarity.py` | **R3-1** 질병 선정/증상 겹침 | `class_symptom_similarity.csv`, `most_confusable_pairs.csv`, `class_similarity_heatmap.png` | ✗ |
| `02_tsne_metrics.py` | **R3-2** t-SNE 정량화 | `tsne_quant_metrics.json`, `tsne_generated.png` | ✗ |
| `03_train.py` | 공용 학습 하니스 | — | ✗ |
| `04_build_ablation.py` | **R4-4** RS/RI 증강셋 생성 | `abl_core/rs/rs_ri/orig_match.csv` | ✗ |
| `05_run_sweep.py` | 스윕 실행(2 GPU) | `results/*.jsonl`, `logs/` | ✗ |
| `06_aggregate.py` | 결과 집계 → 표 | `table4_repro.csv`, `cv_summary.csv`, `ablation_summary.csv` | ✗ |

## 스윕 구성 (총 58 trainings)
- **A. Table 4 재현 + 통계** [R3-4, R4-7, R1-3]: 3 model × 3 batch(8/16/32) × 5 seed = 45 → mean±SD
- **B. 5-fold CV** [R1-3]: klue-bert, 1회 5-fold = 5 trainings
- **C. RS/RI ablation** [R4-4]: klue-bert × 4 arm × 3 seed = 12

## 사용법
```bash
# 1) 즉시 실행되는 분석(학습 불필요)
python3 01_class_similarity.py
python3 02_tsne_metrics.py
python3 04_build_ablation.py --core 1000

# 2) 학습 스윕(백그라운드, 수 시간, resume 지원)
nohup python3 05_run_sweep.py > results/sweep_master.log 2>&1 &

# 3) 진행 상황
cat results/sweep_master.log
tail -f results/logs/<tag>.log

# 4) 완료(또는 부분 완료) 후 집계
python3 06_aggregate.py
```

## 이미 확보된 결과 (학습 무관, 완료)
- **R3-1**: 증상 프로파일 코사인 유사도 상위쌍 — 기관지염-폐렴 0.80, 위염-위장염 0.70, 감기-폐렴 0.70, 외이염-중이염 0.69 등. 리뷰어 지적(증상 겹침)을 정량 확인 → 과제를 "증상 유사 질환 감별 포함 다중라벨 분류"로 재정의하는 근거.
- **R3-2**: 생성 문장 kNN-10 라벨 일치율 = **81.5%**, silhouette=0.05 / 처리전 증상행렬 kNN-10=93.1%, silhouette=0.23. → t-SNE(탐색적) 대신 정량 근거 제시. 경계 중첩은 R3-1의 증상 유사성과 일관.
