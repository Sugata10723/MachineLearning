import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

## 3.5.3 DBCSAN

# データの作成
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# データを平均0分散1にスケール変換
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# DBSCANを実行
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

# 結果をプロット
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, s=20)
plt.xlabel("Feature0")
plt.ylabel("Feature1")
plt.show()
