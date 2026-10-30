import pandas as pd

# 데이터 불러오기
data = pd.read_csv('2025_05_19_data/사용할 데이터/Testset.csv')

# 클래스별로 15개씩 샘플링
sampled_df = data.groupby('labels').sample(n=15, random_state=42)

# labels 기준 정렬
df_sorted = sampled_df.sort_values(by='labels').reset_index(drop=True)

# 저장
output_path = "2025_05_19_data/사용할 데이터/검증데이터/Val_390.csv"
df_sorted.to_csv(output_path, index=False, encoding='utf-8-sig')
