import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


## 3.5.1 k-meansクラスタリング

# 二次元データを作る
x, y = make_blobs(random_state=1)

# クラスタリングモデルを作る
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_, marker='o')
plt.show()
