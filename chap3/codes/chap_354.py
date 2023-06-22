import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

## 顔画像データセットを用いたアルゴリズムの評価
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.image[0].shape

# 画像がジョージ・W・ブッシュに偏っているため、一人当たり50枚に制限する
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# lfwデータから固有顔を抽出し、変換する
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

# デフォルト設定でDBSCANを適用する
dbscsan = DBSCAN()
labels = dbscsan.fit_predict(X_pca)
