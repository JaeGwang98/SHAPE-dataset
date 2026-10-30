import pandas as pd

# 전체 데이터 로드
data = pd.read_csv("2025_05_19_data/4o_mini_total.csv")

# 추출된 데이터 인덱스를 추적하기 위한 집합
selected_indices = set()

origin_data = []
origin_labels = []

index_num = 0
while index_num < len(data):
    for _ in range(200):
        if index_num >= len(data):
            break
        origin_data.append(data['질문'][index_num])
        origin_labels.append(data['labels'][index_num])
        selected_indices.add(index_num)
        index_num += 5
    index_num += 2000

# 원본 데이터프레임에서 제외할 인덱스를 제외한 나머지로 새로운 데이터프레임 생성
remaining_data = data.drop(index=selected_indices)

# 결과 저장
# 5200개 추출 데이터
pd.DataFrame({"질문": origin_data, "labels": origin_labels})\
  .to_csv("2025_05_19_data/4o_mini_origin_5200.csv", index=False, encoding='utf-8-sig')

# 나머지 데이터
remaining_data.to_csv("2025_05_19_data/4o_mini_rest.csv", index=False, encoding='utf-8-sig')
