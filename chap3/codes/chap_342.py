import os 
import pandas as pd
from pandas_profiling import ProfileReport 
from sklearn.decomposition import FactorAnalysis 

# データの読み込み
df_scores = pd.read_csv(os.path.join('..', 'data', '12_subject_scores.csv'))
display(df_scores.head())
display(df_scores.describe())

# profileを見る
Profile = ProfileReport(df_scores, title='Pandas Profiling Report', html={'style': {'full_width':True}})
profile

n_factors =3

fa = FactorAnalysis(n_components=n_factors, random_state=57)
fa.fit(df_scores)

df_factor_loading = pd.DataFrame(fa.components_.T, columns=['factor_{}.format(i)' for i in range(n_factors)], index=df_scores.columns)
df_factor_loading

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
df_samples
factor_scores = fd.transform(df_samples)

pd.DataFrame(factor_scores, colums=['factor_{}'.format(i) for i in range(n_factors)], index=fd_sample.index)
