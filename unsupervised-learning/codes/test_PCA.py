import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import chardet


# 元のCSVファイルを読み込む
data_file = "/Users/ham/Desktop/MachineLearning/unsupervised-learning/data/UNSW_NB15_training-set.csv"  # 元のデータファイルのパスを指定してください
data_df = pd.read_csv(data_file)


# データ標準化 (必要な場合)
scaler = StandardScaler()
X = data_df
X_scaled = scaler.fit_transform(X)

# PCAの適用
n_components = 10  # 削減後の次元数を指定
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# 結果の確認
explained_variance_ratio = pca.explained_variance_ratio_
print(f"各主成分の寄与率: {explained_variance_ratio}")
print(f"累積寄与率: {explained_variance_ratio.sum()}")

# 結果をDataFrameに格納
pca_df = pd.DataFrame(data=X_pca, columns=[f"PC{i+1}" for i in range(n_components)])

# クラスラベルを含む列をデータに結合
result_df = pd.concat([pca_df, merged_df["class_label"]], axis=1)
