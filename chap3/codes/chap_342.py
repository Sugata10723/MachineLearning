import os 
import pandas as pd
from ydata_profiling import ProfileReport 
from sklearn.decomposition import FactorAnalysis 

# データの読み込み
df_scores = pd.read_csv('/Users/ham/Desktop/MachineLearning/chap3/data/12_subject_scores.csv')
print(df_scores.head())
print(df_scores.describe())

# profileを見る
profile = ProfileReport(df_scores, title='Pandas Profiling Report', html={'style': {'full_width':True}})
print(profile)

# factorの設定
n_factors =3

# 因子分析の実行
fa = FactorAnalysis(n_components=n_factors, random_state=57)
fa.fit(df_scores)

# 因子負荷行列の表示
df_factor_loading = pd.DataFrame(fa.components_.T, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_scores.columns)
print(df_factor_loading)

# compute Factor score
df_samples = pd.DataFrame(
    [
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
        [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ],
    columns=df_scores.columns,
    index=['A', 'B', 'C', 'D']
)
print(df_samples)

# 分析の実行
factor_scores = fa.transform(df_samples)

# 因子得点の表示
result = pd.DataFrame(factor_scores, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_samples.index)
print(result)
