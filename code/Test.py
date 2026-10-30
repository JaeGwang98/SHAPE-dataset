import os
# -*- coding: utf-8 -*-
import pandas as pd
import random
from random import shuffle
from openai import OpenAI

# OpenAI 클라이언트 초기화 (API 키는 환경변수로 설정하는 것이 더 안전합니다)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 데이터 로드
data = pd.read_csv("Final_data_0513.csv")
unique_diseases = data['Kor_name'].unique()

# 결과 저장용 리스트
results = []
kor_name = []
# 초기화
disease_count = 0
row_idx = 0
disease_idx = 0

# 증상 조합 데이터 생성
while len(results) < 78000:
    current_name = data.loc[row_idx, 'Kor_name']
    major_symptom = data.loc[row_idx, 'Major']

    # 증상 리스트 구성
    symptoms_present = [col for col in data.columns if data.at[row_idx, col] == 1]
    symptoms_absent = [col for col in data.columns if data.at[row_idx, col] == 0]

    shuffle(symptoms_present)

    # 증상 조합 생성 (원본 + 노이즈)
    for i in range(5):
        symptoms = ", ".join(symptoms_present)
        if i > 0 and symptoms_absent:
            symptoms += ", " + random.choice(symptoms_absent)
        prompt = f"질병명:{current_name}/주요증상:{major_symptom}/현재증상:{symptoms}"
        kor_name.append(current_name)
        results.append(prompt)

    disease_count += 1
    row_idx += 1

    # 같은 질병에서 600개 생성 후 다음 질병으로 전환
    if disease_count == 600:
        # 다음 질병의 인덱스로 이동
        while row_idx < len(data) and data.loc[row_idx, 'Kor_name'] != unique_diseases[(disease_idx + 1) % len(unique_diseases)]:
            row_idx += 1
        disease_count = 0
        disease_idx = (disease_idx + 1) % len(unique_diseases)
    elif row_idx < len(data) and data.loc[row_idx, 'Kor_name'] == unique_diseases[(disease_idx + 1) % len(unique_diseases)]:
        # 질병이 바뀌면 이전 질병으로 다시 돌아감
        row_idx -= disease_count

# 결과 저장
pd.DataFrame({"결과": results}).to_csv("질병_증상_2.csv", index=False, encoding='utf-8-sig')
pd.DataFrame({"label": kor_name}).to_csv("kor_name.csv", index=False, encoding='utf-8-sig')

