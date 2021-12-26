# Machine-Learning

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#day-1">Day 1</a></li>
    <li><a href="#day-2">Day 2</a></li>
    <li><a href="#day-3">Day 3</a></li>
    <li><a href="#day-4">Day 4</a></li>
    <li><a href="#day-5">Day 5</a></li>
    <li><a href="#day-6">Day 6</a></li>
    <li><a href="#day-7">Day 7</a></li>
    <li><a href="#day-8">Day 8</a></li>
    <li><a href="#day-9">Day 9</a></li>
    <li><a href="#day-10">Day 10</a></li>
  </ol>
</details>

## Day 1

IDE : PyCharm 사용

Keras 개발과정

1. 데이터 생성 <br>
2. 모델 구축 model = Sequential(); / model.add(...) <br>
3. 모델 컴파일 model.compile(...) <br>
4. 모델학습 model.fit(...)<br>
5. 모델 평가 및 예측 model.evaluate(...)<br>
6. 모델 저장 model.save(...)<br>

<br><br>
### 데이터 생성 
 train data / validation data / test data<br>
 train data : 학습에 사용되는 데이터이며, 가중치와 바이어스를 최적화하기 위해 사용됨 (during learning)<br>
 validation data : 1 epoch 마다 과적합(overfitting)을 확인하기 위해 사용됨 (during learning)<br>
 test data : 학습 후에 정확도를 평가하거나 임의의 입력에 대한 결과를 예측하기 위해 사용되는 데이터(after learning)

<br><br>
### 모델 구축
 ex) model = Sequential()<br>
     model.add(Flatten(input_shape=(1,)) <br>
     model.add(Dense(2, activation='sigmoid')<br>
     model.add(Dense(1, activation='sigmoid')<br>

<br>

Flatten
 - 입력으로 들어오는 다차원 데이터를 1차원으로 정렬하기 위해 사용되는 레이어이며, 입력 데이터(차원)의 수를 input_shape(1,) 과 같이 기술함

<br>

Dense
 - 각 층의 입력과 출력 사이에 있는 모든 노드가 서로 연결되어 있는 완전 층(FC)을 나타내며, Dense 첫 번째 인자인 2,1등은 출력 노드수를 나타냄<br>

노드의 활성화 함수는 activation="...." 형태로 나타내며, 대표적인 활성화 함수로는 선형회귀 문제에서는 ‘linear’, 일반적인 classification 경우에는 ‘sigmoid’, 'softmax', 'relu', 'tanh' 등이 데이터에 따라 다양하게 사용됨

<br><br>
### ---패션 MNIST 데이터셋 임포트하기(기초적인 분류 문제)--
https://www.tensorflow.org/tutorials/keras/classification?hl=ko

import tensorflow as tf / 텐서플로우 import<br>
from tensorflow import keras / keras API import<br>
import numpy as np / numpy import(helper 라이브러리)<br>
import matplotlib.pyplot as plt / (helper 라이브러리)<br>

fashion_mnist = keras.datasets.fashion_mnist; //keras를 import해 데이터셋을 실험용으로 가져온다.<br>
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data(); //해당 데이터셋의 데이터를 로드한다. 그러면 4개의 Numpy배열이 반환된다.<br>

plt.figure() //매트랩라이브러리에서 모양을 생성해준다.<br>
plt.imshow(train_images[0]) // 해당이미지를 띄워준다.<br>
plt.colorbar() // 밑의 그림에 오른쪽 바생성<br>
plt.grid(False) // 밑의 그림에서 수치마다 줄긋는 UI를 생성할지 말지<br>
plt.show() // 그래프를 띄움<br>

![image](https://user-images.githubusercontent.com/29851990/147409654-a7e5cab0-3886-4fae-adac-285c77734a09.png)
<br>
plt.figure(figsize=(10,10)) //모양의 사이즈를 설정한다<br>
plt.subplot(a,b,i+1) // a는 세로, b는 가로 i+1은 몇번째<br>
plt.xticks([]) //괄호안에 설정하면 x축 좌표를 만들수있음<br>
plt.yticks([]) //괄호안에 설정하면 y축 좌표를 만들수있음<br>
plt.imshow(train_images[i], cmap=plt.cm.binary) // cmap=plt.cm.binary가 회색조로 이미지를 표시하는 것<br>
plt.xlabel(class_names[train_labels[i]]) // 밑의 사진 확대하면 그림밑에 라벨있음<br>
![image](https://user-images.githubusercontent.com/29851990/147409670-a6154280-8f26-40bb-9d7a-fe2c00f89798.png)
<br><br>

<br><br><br><br>
### Sequential Model

Sequential 모델은 각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 일반 레이어 스택에 적합합니다.

<br><br>
---레이어들---<br>
https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/

Keras.layers.Flatten(input_shape=(28, 28)) / 가중치가없고 데이터를 2차원에서 1차원으로 변환만하는 레이어 input_shape은 입력층을 나타냄<br>
keras.layers.Dense(128, activation='relu') / 128개의 출력노드<br>
keras.layers.Dense(10, activation='softmax')

<br>

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])<br>
/--위의 sparse_categorical_crossentropy는 다중 분류 손실함수중 Integer Type의 클래스입니다.

<br>

비슷하게 categorical_crossentropy라는 손실함수가 있는데 이것은 실수 Type의 클래스입니다. 또 mse는 평균제곱오차함수<br>

<br>

metrics
 - 평가기준을 말합니다. 
 - compile() 함수의 metrics 인자로 여러 개의 평가 기준을 지정할 수 있습니다. 
 - 이러한 평가 기준에는 모델의 학습에는 영향을 미치지 않지만, 학습 과정 중에 제대로 학습되고 있는 지 살펴볼 수 있습니다. 
 - 아래는 일반적으로 평가 기준을 ‘accuracy’으로 삽입하였을 때의 코드입니다.

<br>

optimizer (대표적인 것들)
 - SGD(손실함수의 기울기를 계산하여서 이 기울기 값에 학습률을 계산하여 이 결과 값으로 기존의 가중치 값을 갱신하는 확률적 경사하강법)
 - Adam(학습률을 줄여나가고 속도를 계산하여 학습의 갱신강도를 적응적으로 조정해나가는 방법이다.)

model.fit(train_images, train_labels, epochs=5)<br>
/--위의 epochs는 컴파일한 모델을 몇 번 훈련 시키는지입니다.<br>
model.fit에 1,2번째 인자는 입,출력데이터를 넣습니다.

<br>

verbose는 학습중 손실값, 정확도, 진행상태 등의 출력형태를 설정(0,1,2)

<br>

Sequential 모델은 다음의 경우에 적합하지 않습니다.
 - 모델에 다중 입력 또는 다중 출력이 있습니다
 - 레이어에 다중 입력 또는 다중 출력이 있습니다
 - 레이어 공유를 해야 합니다
 - 비선형 토폴로지를 원합니다(예: 잔류 연결, 다중 분기 모델)

<br><br><br><br>

### 용어 정리

Dataset
 - 데이터베이스 자원을 효율적으로 활용하고자 도입된 개념입니다. 
 - 데이터를 추가, 수정, 삭제하거나 연산하는 메소드를 제공하며 다른 Dataset과 병합, 복사하는 등의 기능을 제공합니다.

<br>

MNIST
 - 간단한 컴퓨터 비전 데이터셋입니다.

<br>

Numpy
 - 행렬이나 일반적으로 대규모 다차원 배열을 쉽게 처리 할 수 있도록 지원하는 파이썬의 라이브러리이다.

<br>

Matplotlib
 - 파이썬에서 매트랩과 유사한 그래프 표시를 가능케 하는 라이브러리다.

<br>

Dense Layer
 - 입력과 출력을 모두 연결해주며 입력과 출력을 각각 연결해주는 가중치를 포함하고 있습니다. 
 - 입력이 3개 출력이 4개라면 가중치는 총 3X4인 12개가 존재하게 됩니다. 
 - Dense레이어는 머신러닝의 기본층으로 영상이나 서로 연속적으로 상관관계가 있는 데이터가 아니라면 Dense레이어를 통해 학습시킬 수 있는 데이터가 많다는 뜻이 됩니다.
 - Dense의 첫번째 인자 : 출력 뉴런(노드)의 수를 결정
 - Dense의 두번째 인자 : input_dim은 입력 뉴런(노드)의 수를 결정, 맨 처음 입력층에서만 사용
 - Dense의 세번째 인자 : activation 활성화 함수를 선택

<br>

relu
 - 은닉 층으로 학습
 - 역전파를 통해 좋은 성능이 나오기 때문에 마지막 층이 아니고서야 거의 relu 를 이용한다.

<br>

sigmond
 - yes or no 와 같은 이진 분류 문제

<br>

softmax
 - 확률 값을 이용해 다양한 클래스를 분류하기 위한 문제

<br>

input_dim = 1 
 - 입력 차원이 1이라는 뜻이며 입력 노드가 한개라고 생각하면 됩니다.
 - 만약 x배열의 데이터가 2개라면 2, 3개라면 3으로 지정을 해줍니다.

<br>

loss
 - 손실함수를 의미합니다. 얼마나 입력데이터가 출력데이터와 일치하는지 평가해주는 함수를 의미합니다.
 - ex) https://keras.io/losses/

<br>

optimizer
 - 손실 함수를 기반으로 네트워크가 어떻게 업데이트될지 결정합니다.
 - ex) https://keras.io/ko/optimizers/

<br>

유튜브 Keras강의
 - https://www.youtube.com/watch?v=Ke70Xxj2EJw&t=1223s
