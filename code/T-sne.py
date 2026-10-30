# 필요한 라이브러리 불러오기
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import os

# =======================
# [1] 한글 폰트 설정
# =======================
font_path = "C:/Windows/Fonts/malgun.ttf"  # Malgun Gothic (윈도우)
if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

# =======================
# [2] CSV 파일 불러오기
# =======================
df = pd.read_csv("Final_data_0513.csv")

# =======================
# [3] 숫자형 특성 추출 (증상 벡터)
# =======================
feature_cols = df.select_dtypes(include='int64').columns
X = df[feature_cols]

# =======================
# [4] 라벨 설정 (한글 질병명)
# =======================
labels = df["eng_name"]

# =======================
# [5] PCA로 50차원 축소
# =======================
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)

# =======================
# [6] t-SNE로 2차원 축소
# =======================
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_pca)

# =======================
# [7] t-SNE 결과를 DataFrame으로 정리
# =======================
tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "Kor_name": labels
})

# =======================
# [8] 시각화 (Seaborn으로)
# =======================
plt.figure(figsize=(14, 10))
sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="Kor_name",
    palette="tab20",  # 또는 "hls", "husl"
    s=20,
    alpha=0.7,
    legend="full"
)

plt.title("t-SNE Projection of the Dataset(Processed Data)", fontsize=16)
plt.xlabel("t-SNE_1")
plt.ylabel("t-SNE_2")
plt.grid(True)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title="Disease",
    fontsize=12,
    title_fontsize=13
)
plt.tight_layout()
plt.show()
