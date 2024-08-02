import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm  # Doğru modül

# Veri setini oku
datas = pd.read_csv('odev_tenis.csv')

# Tüm verileri etiket kodlayıcı ile dönüştür
datas2 = datas.apply(preprocessing.LabelEncoder().fit_transform)

# İlk kolonu seç ve OneHotEncoder ile dönüştür
c = datas2.iloc[:, :1]
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

print(c)

# OneHotEncoder sonrası DataFrame oluştur
weather = pd.DataFrame(data=c, index=range(14), columns=['o', 'r', 's'])

# Orijinal veri setinden ilgili kolonları seç ve birleştir
lastdatas = pd.concat([weather, datas.iloc[:, 1:3]], axis=1)
lastdatas = pd.concat([datas2.iloc[:, -2:], lastdatas], axis=1)

print(lastdatas)

# Verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(lastdatas.iloc[:, :-1], lastdatas.iloc[:, -1:], test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

# İstatistiksel model oluşturma ve özet alma
X = np.append(arr=np.ones((14, 1)).astype(int), values=lastdatas.iloc[:, :-1], axis=1)
X_l = lastdatas.iloc[:, [0, 1, 2, 3, 4, 5]].values
r_ols = sm.OLS(endog=lastdatas.iloc[:, -1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

lastdatas = lastdatas.iloc[:, 1:]

X = np.append(arr=np.ones((14, 1)).astype(int), values=lastdatas.iloc[:, :-1], axis=1)
X_l = lastdatas.iloc[:, [0, 1, 2, 3, 4]].values
r_ols = sm.OLS(endog=lastdatas.iloc[:, -1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(y_pred)
