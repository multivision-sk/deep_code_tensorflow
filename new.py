import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist # mnist data set 불러오기

(x_train,y_train), (x_test, y_test) = mnist.load_data()
#mnist dataset 학습용(x,y), 테스트용 (x,y)

x_train.shape
#학습용 데이터 형태


print(x_train[0])
# 학습용 첫번째 데이터

for x in x_train[0] :
    for i in x :
        print('{:3}'.format(i), end='')

    print()

print(y_train[0])
# 학습용 첫번째 데이터 숫자 확인

x_train = x_train/255
x_test = x_test / 255
#데이터 전처리 0~1 숫자로

print(x_train[0])
#데이터 전처리 결과 확인

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(256,activation = 'relu'),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax') #확률 출력

])
#모델 만들기 : 입력층(784) - 은닉층1(256), 은닉층2(128) - 은닉층3(64) - 출력층(10)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics =['accuracy'])
#모델 컴파일 : 최적화, 손실함수, 평가지표

print(model.summary())

model.fit(x_train,y_train, epochs = 5)
#모델 학습, 전체 데이터는 5번 반복

model.evaluate(x_test, y_test)
#모델평가 , 처음본 데이터에 대해 평가

plt.imshow(x_train[0], cmap='gray')
plt.show()
#예측 - 0번째 숫자 이미지로

print(np.argmax(model.predict(x_train[0].reshape(1,28,28))))
#예측 - 0번째 숫자 예측




