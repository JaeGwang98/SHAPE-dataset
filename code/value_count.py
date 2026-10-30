import pandas as pd

data = pd.read_csv("data_after_test.csv")
disease_name = data['diseases'].unique()
print(len(disease_name))
# print(data['diseases'].value_counts()['cardiac arrest'])

low_frequency_diseases = set(data['diseases'].value_counts().loc[lambda x: x > 300].index)

with open('disease_list_ko.txt', 'r', encoding='utf-8') as f:
    filtered_items_2 = {item.strip() for item in f.read().split('\n') if item.strip()}


print(low_frequency_diseases)
print(len(filtered_items_2))