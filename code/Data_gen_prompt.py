# -*- coding: utf-8 -*-
import random
from copy import deepcopy
import json
import pandas as pd

# CSV 파일 읽기 (한글 깨짐 방지)
data = pd.read_csv("origin_5200.csv", encoding="utf-8-sig")

# 결측치 제거 (빈 값이 있는 경우 대비)
data = data.dropna(subset=["결과"])
example_data = pd.read_csv('4o_example.csv')


init_template = {
    "custom_id": None,
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o",
        "messages": None  # 후에 삽입할 예정
    }
}

# 각 row마다 입력 구성
batches = []
for idx, query in enumerate(data["결과"]):
    random_example_num_1, random_example_num_2, random_example_num_3 = random.sample(range(0,100),3)
    prompt_example_1, prompt_example_2, prompt_example_3 = example_data['result'][random_example_num_1], example_data['result'][random_example_num_2], example_data['result'][random_example_num_3]

    system_prompt = (
        "당신은 주어진 질병과 증상을 겪고 있는 환자입니다. 주어진 증상에 관하여 의사에게 질문을 하세요.\n"
        "질문에 질병명을 포함하지 않고 자연스러운 문장으로 구성하세요.\n"
        "임의의 상황(완치 내역, 증상이 발생한 날짜, 나이, 성별)을 부여하여 구성하세요.\n"
        "모든 문장은 한국어로 작성합니다.\n"
        "예시 3가지를 제공합니다. 다음 예시 3가지를 참고하여 질문 1개를 구성하세요.\n"
        f"{prompt_example_1}\n "
        f"{prompt_example_2}\n"
        f"{prompt_example_3}\n"

    )
    temp = deepcopy(init_template)
    temp["custom_id"] = str(idx)
    temp["body"]["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    batches.append(temp)

# JSONL 파일로 저장
with open("batch_4o_origin_random_fewshot.jsonl", "w", encoding="utf-8") as f:
    for item in batches:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")
