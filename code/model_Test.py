import os
# -*- coding: utf-8 -*-
import pandas as pd
import random
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#example_data_load
example_data = pd.read_csv("4o_example.csv")

# 데이터 로드
data = pd.read_csv("kor_name.csv")

# system_prompt = (
#     "당신은 주어진 질병과 증상을 겪고 있는 환자입니다. 주어진 증상에 관하여 의사에게 질문을 하세요.\n"
#     "질문에 질병명을 포함하지 않고 자연스러운 문장으로 구성하세요.\n"
#     "임의의 상황(완치내역, 증상이 발생한 날짜, 나이, 성별)을 부여하여 구성하세요.\n"
#     "모든 문장은 한국어로 작성합니다.\n"
#     "예시 3가지를 제공합니다. 다음 예시 3가지를 참고하여 질문 1개를 구성하세요.\n"
#     "일주일 전부터 가슴이 약간 아프고, 특히 명치 끝에서 목구멍 쪽으로 치밀어 오르는 듯한 타는 느낌이 계속됩니다.음식을 먹고 난 후에 증상이 심해지는 것 같고, 누워 있을 때도 불편함이 느껴집니다.가끔 목이 화끈거리는 것 같고, 속이 더부룩하고 가벼운 구역질이 나는 경우도 있습니다.평소보다 소화가 잘 안 되는 것 같고, 기름진 음식, 매운 음식을 먹은 후 증상이 더 심해지는 것 같아요.\n"
#     "며칠 전부터 치주 부위가 붓고, 입술도 부어 오르는 느낌이 듭니다. 양치할 때 잇몸에서 피가 나는 건 물론이고, 귀 쪽으로도 통증이 가는 것 같아요. 잇몸과 치아 쪽의 고통이 점점 심해지고 있는데, 어떻게 하면 통증을 완화시킬 수 있을까요?\n"
#     "작년에 당뇨병 완치를 받았었는데 목이 자주 마르고 물을 많이 마셔도 계속 갈증이 나며, 체중이 갑자기 줄고 피로감이 느껴집니다."
# )


system_prompt = ("주어진 질병 관해서 환자들이 의사들에게 할만한 질문을 1가지 작성해줘. 질병명을 질문에 포함하지 않고 증상 위주로 작성해주세요.\n"
                 "예시\n"
                 "1. 요 며칠 사이에 갑자기 숨 쉬기가 불편해졌어요. 특히 깊게 숨을 들이쉴 때 가슴이 꽉 조이는 느낌이 들고, 가끔은 숨 쉴 때 통증도 있어요. 특별히 운동을 많이 한 것도 아닌데 이런 증상이 왜 생기는 건지 걱정돼서 왔어요.\n"
                 "2. 그냥 평소처럼 생활하는데도 숨이 차고, 계단 몇 층만 올라가도 숨이 턱 막히는 것 같아요. 원래 이런 증상은 없었는데, 최근 감기처럼 열이 좀 나고 기침을 한 뒤부터 이렇게 된 것 같아요. 혹시 폐나 호흡기 쪽에 이상이 있는 건 아닐까요\n"
                 "3. 기침이 계속 나고 멈추질 않아요. 처음에는 그냥 마른기침이었는데, 요즘은 가래가 끼고 가끔 누런색이나 녹색빛이 도는 가래가 나오기도 해요. 밤에 특히 심해서 잠을 제대로 못 자고 있고요. 감기치고는 좀 오래가는 것 같아서 걱정돼요.")

csv_result = []

for i in range(100):
    random_num = random.randint(1, 78000)
    # random_example_num_1, random_example_num_2, random_example_num_3 = random.sample(range(0,100),3)
    prompt = data['label'][random_num]
    # example_1, example_2, example_3 = example_data['result'][random_example_num_1], example_data['result'][random_example_num_2], example_data['result'][random_example_num_3]

    # system_prompt = (
    #     "당신은 주어진 질병과 증상을 겪고 있는 환자입니다. 주어진 증상에 관하여 의사에게 질문을 하세요.\n"
    #     "질문에 질병명을 포함하지 않고 자연스러운 문장으로 구성하세요.\n"
    #     "임의의 상황(완치 내역, 증상이 발생한 날짜, 나이, 성별)을 부여하여 구성하세요.\n"
    #     "모든 문장은 한국어로 작성합니다.\n"
    #     "예시 3가지를 제공합니다. 다음 예시 3가지를 참고하여 질문 1개를 구성하세요.\n"
    #     f"{example_1}\n "
    #     f"{example_2}\n"
    #     f"{example_3}\n"
    #
    # )

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

    csv_result.append({"result": result})

df = pd.DataFrame(csv_result)
# df.to_csv("4.1-mini_example.csv", encoding='utf-8-sig', index=False)