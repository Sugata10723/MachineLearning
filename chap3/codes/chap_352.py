import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering


## 3.5.2 凝集型クラスタリング
X, y = make_blobs(random_state=100)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=assignment, marker='o')
plt.xlabel("Feature0")
plt.ylabel("Feature1")
plt.show()
