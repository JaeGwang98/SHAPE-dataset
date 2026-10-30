import os
# -*- coding: utf-8 -*-
import pandas as pd
import random
from openai import OpenAI

# OpenAI 클라이언트 초기화 (API 키는 환경변수로 설정하는 것이 더 안전합니다)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 데이터 로드
data = pd.read_csv("kor_name.csv")
disease_name = data['label'].unique()

system_prompt = ("주어진 질병 걸린 환자들이 의사들에게 할만한 질문을 1가지 작성해주세요. 질병명을 질문에 포함하지 않고 증상 위주로 작성해주세요.\n"
                 "예시\n"
                 "1. 요 며칠 사이에 갑자기 숨 쉬기가 불편해졌어요. 특히 깊게 숨을 들이쉴 때 가슴이 꽉 조이는 느낌이 들고, 가끔은 숨 쉴 때 통증도 있어요. 특별히 운동을 많이 한 것도 아닌데 이런 증상이 왜 생기는 건지 걱정돼서 왔어요.\n"
                 "2. 그냥 평소처럼 생활하는데도 숨이 차고, 계단 몇 층만 올라가도 숨이 턱 막히는 것 같아요. 원래 이런 증상은 없었는데, 최근 감기처럼 열이 좀 나고 기침을 한 뒤부터 이렇게 된 것 같아요. 혹시 폐나 호흡기 쪽에 이상이 있는 건 아닐까요\n"
                 "3. 기침이 계속 나고 멈추질 않아요. 처음에는 그냥 마른기침이었는데, 요즘은 가래가 끼고 가끔 누런색이나 녹색빛이 도는 가래가 나오기도 해요. 밤에 특히 심해서 잠을 제대로 못 자고 있고요. 감기치고는 좀 오래가는 것 같아서 걱정돼요.")

csv_result = []

for i in range(200):
    random_num = random.randint(0, len(disease_name)-1)
    prompt = disease_name[random_num]

    completion = client.chat.completions.create(
                model="gpt-4o",
                store=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"질병명:{prompt}"}
                ]
            )
    result = completion.choices[0].message.content
    print(result + "\n")

    csv_result.append({"question": result, "disease": prompt})

df = pd.DataFrame(csv_result)
df.to_csv("Verification_data_200.csv", encoding='utf-8-sig', index=False)