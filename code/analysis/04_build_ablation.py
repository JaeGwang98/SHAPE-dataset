# -*- coding: utf-8 -*-
"""
Reviewer 4-4 대응: RS/RI(EDA) 증강 효과 검증용 학습셋 생성 (API 불필요).
근거: 논문이 사용한 RS(random swap)/RI(random insertion)를 생성된 한국어 문장에
      직접 적용(Wei & Zou EDA, 본 논문 참고문헌 [17])하여 증강 유무를 통제 비교.

생성 arm (모두 Trainset에서 클래스당 CORE개를 코어로 사용):
  A. abl_core.csv       : 코어만 (증강 없음)              CORE/class
  B. abl_rs.csv         : 코어 + RS 사본                 2*CORE/class
  C. abl_rs_ri.csv      : 코어 + RS 사본 + RS+RI 사본     3*CORE/class
  D. abl_orig_match.csv : 원본만 3*CORE/class (크기 대조군, 증강 대신 실제 문장 추가)

비교 논리:
  C vs A  -> 증강이 성능을 높이는가
  C vs D  -> 증강이 '같은 양의 실제 문장 추가'만큼/이상 기여하는가 (데이터량 교란 통제)
"""
import argparse, random
import pandas as pd

DATA = "../2025_05_19_data/사용할 데이터"
TRAIN_CSV = f"{DATA}/25.05.19/Trainset.csv"
OUT = "results"

def random_swap(words, n):
    words = words[:]
    L = len(words)
    if L < 2:
        return words
    for _ in range(n):
        i, j = random.sample(range(L), 2)
        words[i], words[j] = words[j], words[i]
    return words

def random_insertion(words, n):
    words = words[:]
    if not words:
        return words
    for _ in range(n):
        w = random.choice(words)          # 문장 내 임의 어절을 임의 위치에 삽입 (EDA RI)
        words.insert(random.randint(0, len(words)), w)
    return words

def aug(text, use_ri, alpha=0.1):
    words = str(text).split()
    n = max(1, int(alpha * len(words)))
    w = random_swap(words, n)
    if use_ri:
        w = random_insertion(w, n)
    return " ".join(w)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=int, default=1000, help="클래스당 코어 문장 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    tr = pd.read_csv(TRAIN_CSV)
    core = tr.groupby("labels", group_keys=False).apply(
        lambda g: g.sample(n=min(args.core, len(g)), random_state=args.seed)
    ).reset_index(drop=True)
    match = tr.groupby("labels", group_keys=False).apply(
        lambda g: g.sample(n=min(3 * args.core, len(g)), random_state=args.seed + 1)
    ).reset_index(drop=True)

    # A. core
    core.to_csv(f"{OUT}/abl_core.csv", index=False, encoding="utf-8-sig")

    # B. core + RS
    rs = core.copy(); rs["질문"] = rs["질문"].map(lambda t: aug(t, use_ri=False))
    pd.concat([core, rs], ignore_index=True).to_csv(
        f"{OUT}/abl_rs.csv", index=False, encoding="utf-8-sig")

    # C. core + RS + (RS+RI)
    rsri = core.copy(); rsri["질문"] = rsri["질문"].map(lambda t: aug(t, use_ri=True))
    pd.concat([core, rs, rsri], ignore_index=True).to_csv(
        f"{OUT}/abl_rs_ri.csv", index=False, encoding="utf-8-sig")

    # D. 크기 대조군(원본 3x)
    match.to_csv(f"{OUT}/abl_orig_match.csv", index=False, encoding="utf-8-sig")

    for name, n in [("abl_core", len(core)), ("abl_rs", 2*len(core)),
                    ("abl_rs_ri", 3*len(core)), ("abl_orig_match", len(match))]:
        print(f"{name}: {n} rows")
    print("\n예시(원본 -> RS -> RS+RI):")
    s = core["질문"].iloc[0]
    print(" 원본 :", s[:80])
    print(" RS   :", aug(s, use_ri=False)[:80])
    print(" RS+RI:", aug(s, use_ri=True)[:80])

if __name__ == "__main__":
    main()
