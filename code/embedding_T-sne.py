# 필요한 라이브러리 설치 (Colab/로컬에서 처음 실행 시)
# !pip install sentence-transformers matplotlib seaborn scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import os
from matplotlib import font_manager, rc

# ✅ 한글 폰트 설정 (Malgun Gothic)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

# 📥 CSV 파일 로드
df = pd.read_csv("2025_05_19_data/사용할 데이터/25.06.04/total_dataset.csv")

# 🧠 Sentence-BERT 모델 로드
print("👉 문장 임베딩 중...")
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
embeddings = model.encode(df["질문"].tolist(), show_progress_bar=True)

# 🔻 TSNE 차원 축소
print("👉 t-SNE 차원 축소 중...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced = tsne.fit_transform(embeddings)

# 📊 시각화용 DataFrame
tsne_df = pd.DataFrame({
    "TSNE-1": reduced[:, 0],
    "TSNE-2": reduced[:, 1],
    "Label": df["eng_name"]
})

# 🎨 시각화
plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=tsne_df,
    x="TSNE-1", y="TSNE-2",
    hue="Label",
    palette="tab10",
    s=15,
    linewidth=0,
    alpha=0.7,
    legend='full'
)
plt.title("t-SNE Projection of the Dataset(Symptom–Disease Dataset) ", fontsize=16)
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title="Disease",
    fontsize=12,
    title_fontsize=13
)
plt.tight_layout()
plt.show()
