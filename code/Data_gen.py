import os
# -*- coding: utf-8 -*-
import pandas as pd
from openai import OpenAI
from random import shuffle
import random

#api_key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#data_reading
data = pd.read_csv("data_after_test.csv")
#한국어_data_reading
with open('disease_list_ko.txt', 'r', encoding='utf-8') as f:
    filtered_items = [item.strip() for item in f.read().split('\n') if item.strip()]
#질병명_중복제거
disease_name = data['diseases'].unique()

results = []

plus_num = 0

for num in range(8408, 8418):
    random_int = random.randint(0, 1)
    symptoms_list = []
    #질병명_한국어_영어_매칭
    disease = filtered_items[list(disease_name).index(data['diseases'][num])]
    #증상 중 1인것만 출력
    for col in data.columns:
        if data[col][num] == 1:
            symptoms_list.append(col)
    if random_int == 1:
    #증상 중 0인것도 넣어서 데이터 증강
        random_num = random.randint(1, len(data.columns)-1)
        if data[data.columns[random_num]][num] == 0:
            symptoms_list.append(data.columns[random_num])
            plus_num += 1
    #순서 섞어서 데이터 증강
    shuffle(symptoms_list)
    symptoms = ", ".join([col for col in symptoms_list])
    prompt = f"질병명:{disease} / 증상:{symptoms}"
    #GPT_API로 전달하여 데이터 재구성
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            store=True,
            messages=[
                {"role": "system", "content": (
                    "주어진 질병명과 증상을 활용하여, 환자가 할만한 질문을 한국어로 작성해주세요.\n"
                    "'질문'에 질병명을 포함하지 않으며, '최근에'라는 단어를 사용하지 않고 임의의 상황을 부여하여 자연스러운 문장으로 구성하세요.\n"
                    "예시 3가지를 제공합니다. 예시 형식에 맞추어 1가지 질문만 작성해주세요.\n"
                    "'질문':일주일 전부터 가슴이 약간 아프고, 특히 명치 끝에서 목구멍 쪽으로 치밀어 오르는 듯한 타는 느낌이 계속됩니다. 음식을 먹고 난 후에 증상이 심해지는 것 같고, 누워 있을 때도 불편함이 느껴집니다. 가끔 목이 화끈거리는 것 같고, 속이 더부룩하고 가벼운 구역질이 나는 경우도 있습니다. 평소보다 소화가 잘 안 되는 것 같고, 기름진 음식, 매운 음식을 먹은 후 증상이 더 심해지는 것 같아요. \n"
                    "'질문':며칠 전부터 치주 부위가 붓고, 입술도 부어 오르는 느낌이 듭니다. 양치할 때 잇몸에서 피가 나는 건 물론이고, 귀 쪽으로도 통증이 가는 것 같아요. 잇몸과 치아 쪽의 고통이 점점 심해지고 있는데, 어떻게 하면 통증을 완화시킬 수 있을까요? \n"
                    "'질문':작년에 당뇨병 완치를 받았었는데 목이 자주 마르고 물을 많이 마셔도 계속 갈증이 나며, 체중이 갑자기 줄고 피로감이 느껴집니다.\n"
                )},
                {"role": "user", "content": prompt}
            ]
        )
        print(prompt)
        result = completion.choices[0].message.content
        print(result+"\n")
        question = result.split("'질문':")[1].strip()
        #질문과 질병명 나누어 저장
        results.append({"질문": question, "질병명": disease, "Prompt": prompt})



    except Exception as e:
        print(f"Error at row {num}: {e}")
        results.append({"질문": result, "질병명": disease, "Prompt": prompt})

print(plus_num)
#저장
df = pd.DataFrame(results)
df.to_csv("data_300_11_4o.csv", index=False, encoding="utf-8-sig")
print("저장 완료")
