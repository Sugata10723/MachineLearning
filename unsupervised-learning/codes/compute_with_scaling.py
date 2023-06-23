from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

## 教師あり学習における前処理の効果

cancer = load_breast_cancer()
scaler = MinMaxScaler()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# 前処理なし
svm = SVC(C=100)
svm.fit(x_train, y_train)
print("Test set sccuracy: {:.2f}".format(svm.score(x_test, y_test))) # 何故かめちゃくちゃ精度がいい

# 前処理あり
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
svm.fit(x_train_scaled, y_train)
print("Scaled test set sccuracy: {:.2f}".format(svm.score(x_test_scaled, y_test)))
