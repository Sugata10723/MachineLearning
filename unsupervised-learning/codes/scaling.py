import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## 3.3.2 データ変換の適用
cancer = load_breast_cancer()
scaler = MinMaxScaler()

# データの分割
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

# x_trainとx_testに分割させ、その数を表示
print(x_train.shape)
print(x_test.shape)

# スケール変換器のfitメソッド
scaler.fit(x_train)
MinMaxScaler(copy=True, feature_range=(0,1))

# データを変換
x_train_scaled = scaler.transform(x_train)

# スケール変換の前後のデータ特性をプリント
plt.plot(x_train_scaled)

# テストデータを変換
x_test_scaled = scaler.transform(x_test)

# 変換したテストデータをプロット
print(x_test_scaled[:,0]) # 第一成分だけで表示
plt.plot(x_test_scaled) 
plt.show() # この時、MinMaxScalerが常に訓練データとテストデータに全く同じ変換を施すので、テストデータの最大最小が0と1にならない
