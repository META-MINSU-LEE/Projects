# -*- coding: utf-8 -*-
"""project4 IV.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rePJmyXjmo3pP6SGJDh07jAo_JWusJsp
"""

!git clone https://github.com/META-MINSU-LEE/Projects



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np


# MEPFIL USP1 데이터를 불러옵니다.
df = pd.read_csv('./Projects/DATA/MEPFILDUSP1.csv')

# 데이터 값을 살펴보겠습니다.
df

# 데이터가 어떤 유형으로 이루어져 있는지 알아봅니다.
df.dtypes

# 속성별로 결측치가 몇 개인지 확인합니다.
df.isnull().sum().sort_values(ascending=False).head(12)

# 데이터 사이의 상관 관계를 저장합니다.
df_corr=df.corr()

# 4W와 관련이 큰 것부터 순서대로 저장합니다.
df_corr_sort=df_corr.sort_values('4W Avg', ascending=False)

# 4W와 관련도가 가장 큰 10개의 속성들을 출력합니다.
df_corr_sort['4W Avg'].head(12)

# 4W와 관련도가 가장 높은 속성들을 추출해서 상관도 그래프를 그려봅니다.
cols=['Straight','2W','EP Avg','Diameter']
sns.pairplot(df[cols])
plt.show();



# 4W 값을 제외한 나머지 열을 저장합니다.
cols_train=['Straight','2W','EP Avg','Diameter']
X_train_pre = df[cols_train]

# 4W 값을 저장합니다.
y = df['4W Avg'].values

# 전체의 80%를 학습셋으로, 20%를 테스트셋으로 지정합니다.
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)

# 모델의 구조를 설정합니다.
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

# 모델을 실행합니다.
model.compile(optimizer ='adam', loss = 'mean_squared_error')

# 20회 이상 결과가 향상되지 않으면 자동으로 중단되게끔 합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30)

# 모델의 이름을 정합니다.
modelpath="./data/model/TEST.TEST2"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# 실행 관련 설정을 하는 부분입니다. 전체의 20%를 검증셋으로 설정합니다.
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])

# 예측 값과 실제 값, 실행 번호가 들어갈 빈 리스트를 만듭니다.
real_4WAvg = []
pred_4WAvg = []
X_num = []

# 25개의 샘플을 뽑아 실제 값, 예측 값을 출력해 봅니다.
n_iter = 0
Y_prediction = model.predict(X_test).flatten()
for i in range(25):
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제4W Avg: {:.2f}, 예상4W Avg: {:.2f}".format(real, prediction))
    real_4WAvg.append(real)
    pred_4WAvg.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)

# 그래프를 통해 샘플로 뽑은 25개의 값을 비교해 봅니다.

plt.plot(X_num, pred_4WAvg, label='predicted 4W Avg')
plt.plot(X_num, real_4WAvg, label='real 4W Avg')
plt.legend()
plt.show()

