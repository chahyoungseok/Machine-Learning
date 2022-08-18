# Machine-Learning

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#keras-development">Keras Development</a></li>
    <li><a href="#sequential-model">Sequential Model</a></li>
    <li><a href="#glossary">Glossary</a></li>
    <li><a href="#regression">Regression</a></li>
    <li><a href="#perceptron">Perceptron</a></li>
    <li><a href="#activation-function">Activation Function</a></li>
    <li><a href="#loss-function">Loss Function</a></li>
    <li><a href="#optimizer">Optimizer</a></li>
    <li><a href="#overfit">Overfit</a></li>
    <li><a href="#cnn">CNN</a></li>
    <li><a href="https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10664524">연구논문-링크</a></li>
  </ol>
</details>

## Keras Development

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

### keras development example

``` keras development example
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

real_input = np.array(
        [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
         [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]])

real_output = np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],
                       [0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],
                       [0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],
                       [0,0,0,0,0,0,0,0,0,1]])


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu',input_shape=(10,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['accuracy'])

model.fit(real_input*np.random.uniform(0.8,0.9) + np.random.uniform(0,0.1), real_output, epochs=10000)

#input_patten2
test_input = np.array([[0, 0.9, 0, 0.91, 0, 0.99, 0, 0.92, 0, 1]])
test_output = np.array([[0,1,0,0,0,0,0,0,0,0]])
prediction = model.predict(test_input)
print(prediction)

model.save("PattenTrain.h5")

```

#### 데이터 셋 <br>
입력으로 10개의 패턴을 가진 배열을 생성합니다. : real_input <br>
출력으로는 10개의 노드에 해당 패턴이 mapping될 수 있게 배열을 생성합니다 : real_output

#### 모델 생성 <br>
모델은 2차원배열을 1차원배열로 바꿀 이유가없으니 Flatten을 사용하지 않습니다.<br>
Dense layer를 사용하였고, 후에 predict을 통해 분포를 확인하기위해 softmax를 사용하였습니다.

#### 모델훈련 <br>
모델을 학습시키려면 방대한양의 데이터가 필요하므로 기존에 입력데이터에 랜덤값(0.8, 0.9)을 곱한 후, 랜덤 값(0,0.1)을 더하고 모델훈련을 시킵니다.

#### 모델 예측
테스트 데이터로는 test_input을 만들어 기존 패턴과 근사하게 만들어 predict해봅니다.

#### 모델 저장 및 로드
model.save("PattenTrain.h5")를 통해 모델을 저장시키고, 새로운 파일 에서 keras.models.load_model("PattenTrain.h5")을 통해 모델을 가져와 summary와 perdict을 해보아 모델이 제대로 불러와 졌는지 확인하였습니다.

![image](https://user-images.githubusercontent.com/29851990/147450584-7c568169-6183-4c7b-b1c0-8bb4ae9ca90f.png)


<br><br><br><br>
## Sequential model

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

## Glossary

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

<br><br><br><br>

## Regression

회귀모델
 - 연속성 데이터 y와 이것의 원인이 되는 x와의 관계를 추정하기 위해 만든 형식
 - y = f(x) + e (단 e는 실제 환경적인 오차 or 잡음)

회귀분석
 - 인과관계를 수학적으로 분석하는 것
 - 종속변수 y와 독립변수 x의 관계에 따라서 선형, 비선형으로 나뉘게 됩니다

<br>

### linear regreesion or non-linear regression
 - 1. Function
 - 2. Single Layer or Multi Layer
 - 3. Activation Function

<br>

구분하는방법
 - Regression의 linear 유무는 설명변수 x에 대해서 linear한지 안한지로 구분하지않습니다.
 - 식이 갖는 파라미터(회귀변수)에 대하여 linear한지 안한지로 구분합니다.

<br>

#### 1. Function (activation Function이 y=x라고 가정합니다.)

a,b에 대하여 linear하므로 linear regreesion이라 부를수 있습니다.

![1](https://user-images.githubusercontent.com/29851990/152635221-5f943ddb-e98f-4af0-8d40-21d88d68f602.PNG)

a,b,c에 대하여 linear하므로 linear regreesion이라 부를수 있습니다.

![2](https://user-images.githubusercontent.com/29851990/152635463-3b162c75-06f2-43ba-af2b-3b28e5f88375.PNG)

양변에 자연로그 ln을 취해주면 ln(y) = ln(a) + bln(x)에서 y_tilda = a_tilda + b * x_tilda이므로 linear regreesion으로 볼 수 있습니다.

![3](https://user-images.githubusercontent.com/29851990/152635468-ad18edc1-7eab-422c-a36d-aef5ce20f375.PNG)

양변에  (1 - y)로 나누어주면 ln(y / 1 - y) = ax^2 + bx + c가 되므로 linear regreesion으로 볼 수 있습니다.

![4](https://user-images.githubusercontent.com/29851990/152635548-da5b4cbf-46ef-4365-8409-aa5db3c6c272.PNG)

아래와 같은 경우의 함수는 linear regreesion을 사용할 수 없습니다.

![5](https://user-images.githubusercontent.com/29851990/152635690-51f11fa3-a45f-480d-b705-82ebf85ab856.PNG)

<br>

#### 2. Single Layer or Multi Layer

아래의 사진은 Multi Layer의 출력을 나타낸것입니다. <br>
최종결과는 결국 x1앞의 계수를 w로 치환하며 linearizable 해집니다.<br>
따라서 Multi Layer라고 하여 non-linear regression이라는 표현은 맞지않습니다.

![6](https://user-images.githubusercontent.com/29851990/152636126-8ad1f0da-921a-4d1d-b7a1-6b87673618df.PNG)

#### 3. Activation Function

결과론 적으로 activation Function이 linear와 non-linear를 결정짓는데 가장 중요한 역할을 합니다. <br>
1번과 2번을 자세히 보면 알겠지만 activation function이 y = x라면 Multi Layer여도 linear합니다. <br>
하지만 activation funcation이 non-linear한 function을 쓰게된다면 보편적으론 파라미터들에 대해서는 linear해질 수 없습니다. <br>


<br><br><br><br>

## Perceptron

퍼셉트론 
 - 다수의 입력을 받아 하나의 신호를 출력하는 알고리즘

해당 식 
 - x1*w1 + x2*w2 + b(bias) = y

![image](https://user-images.githubusercontent.com/29851990/147449395-0b5d896d-08cf-49bd-bbff-fdb65d2a6c65.png)

위의 식을 활용해 평면상의 선을 그어 적합한 기울기를 찾아냅니다.<br>
하지만 XOR 게이트는 밑의 그림처럼 하나의 선으로는 분류하기 불가능하여 다중퍼셉트론을 사용합니다.

![image](https://user-images.githubusercontent.com/29851990/147449425-2e0658fc-c699-45ac-adee-ee67ed265e0b.png)
<br>


### Perceptron Example
``` python

import numpy as np

# inputData01에서 0과 1을 적절히 조합할만한 배열을 만듭니다.
inputData01 = np.array([0,0,1,1,0,1,0,1]).reshape(2,4)

# inputData02에서 0과 1을 적절히 조합할만한 배열을 만듭니다.
inputData02 = np.array([1,1,-1,-1,1,-1,1,-1]).reshape(2,4)

# 퍼셉트론이란 다수의 입력을 받아 특정알고리즘을 바탕으로 하나의 신호를 출력하여 그결과에 따라 흐른다(1), 안흐른다(0)으로 출력하는 알고리즘을 말합니다.
# 단층퍼셉트론으로는 XOR게이트를 만들 수 없으므로 다중퍼셉트론을 구현합니다.
# 0,0 / 1,1 일때와 1,0 / 0,1일때로 나눈 결과값을 만들수 있는 함수를 만들어줍니다.
# perceptron1,2,3은 inputData01에 해당하는 perceptron
# perceptron4,5,6은 inputData02에 해당하는 perceptron

# (0,0)이 아니라면 무조건 1반환
def perceptron1 (input1, input2) :
    weight1 = 1
    weight2 = 1
    bias = 0
    result = (weight1 * input1) + (weight2 * input2) + bias

    if(result > 0) :
        return 1
    else :
        return 0

# (1,1)이 아니라면 무조건 0반환
def perceptron2(input1, input2):
    weight1 = 1
    weight2 = 1
    bias = -1
    result = (weight1 * input1) + (weight2 * input2) + bias

    if (result > 0):
        return 1
    else:
        return 0

# 같으면 0, 다르면 1이되게 함수구현 다를때는 무조건 1,0이 perceptron3함수에 인자로 전달되게끔 perceptron1,2를 설계
def perceptron3(input1, input2):
    weight1 = 1
    weight2 = -2
    bias = 0
    result = (weight1 * input1) + (weight2 * input2) + bias

    if (result > 0):
        return 1
    else:
        return 0

# (-1,-1)이아니라면 무조건 1반환
def perceptron4 (input1, input2) :
    weight1 = 1
    weight2 = 1
    bias = 2
    result = (weight1 * input1) + (weight2 * input2) + bias

    if(result > 0) :
        return 1
    else :
        return 0

# (1,1) 이아니라면 무조건 0반환
def perceptron5(input1, input2):
    weight1 = 1
    weight2 = 1
    bias = -1
    result = (weight1 * input1) + (weight2 * input2) + bias

    if (result > 0):
        return 1
    else:
        return 0

# 같으면 0, 다르면 1이되게 함수구현 다를때는 무조건 1,0이 perceptron6함수에 인자로 전달되게끔 perceptron4,5를 설계
def perceptron6(input1, input2):
    weight1 = 1
    weight2 = -2
    bias = 0
    result = (weight1 * input1) + (weight2 * input2) + bias

    if (result > 0):
        return 1
    else:
        return 0

for i in range(0,4) :
    print("입력 :" + str(inputData01[0,i]) + "," + str(inputData01[1,i]) + " 출력 :" + str(perceptron3(perceptron1(inputData01[0,i],inputData01[1,i]),perceptron2(inputData01[0,i],inputData01[1,i]))))

for i in range(0,4) :
    print("입력 :" + str(inputData02[0,i]) + "," + str(inputData02[1,i]) + " 출력 :" + str(perceptron6(perceptron4(inputData02[0,i],inputData02[1,i]),perceptron5(inputData02[0,i],inputData02[1,i]))))
```

하지만 다중퍼셉트론을 사용하면 가중치와 바이어스를 하나하나 설정해야 되므로 이것을 학습시키기 위해서 backpropagation을 사용합니다.

```python

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) :
    return 1/(1+np.exp(-x))

x = np.array([[1,1],[1,0],[0,1],[0,0]])
y = np.array([[0],[1],[1],[0]])

#-1,1을 사용하려면 위에것을 주석 후 밑의 문장 주석해제
#x = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
#y = np.array([[-1],[1],[1],[-1]])


#모델 구성하기 입력층 노드2개와 은닉층 2개의 노드 또, 출력층 1개의 노드이므로 이렇게 구성하였습니다. (units, input_shape)
#activation Function은 미분가능한 0,1을 출력하는 sigmoid함수를 사용하였습니다.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

#Gradient Descent 와 SGD의 차이점은 전자는 전체데이터를 가지고 수렴을시키고
#SGD는 일부데이터를 가지고 빠르게 수렴을 시킨다.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mse')

#batch_size는 데이터셋을 나눌때의 크기를 의미합니다.
history = model.fit(x,y,epochs=15000,batch_size=1)

prediction = model.predict(x)

print("예측 Output : ", prediction)

plt.figure()
plt.grid(True)
weight_ary = model.get_weights()

print(weight_ary)

plt.scatter(weight_ary[0][0][0],weight_ary[1][0],color="r",edgecolors="r")
plt.scatter(weight_ary[0][0][1],weight_ary[1][0],color="r",edgecolors="r")

plt.scatter(weight_ary[0][1][0],weight_ary[1][1],color="b",edgecolors="b")
plt.scatter(weight_ary[0][1][1],weight_ary[1][1],color="b",edgecolors="b")

plt.scatter(weight_ary[2][0][0],weight_ary[3][0],color="g",edgecolors="g")
plt.scatter(weight_ary[2][1][0],weight_ary[3][0],color="g",edgecolors="g")

plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10])
plt.yticks([-10,-8,-6,-4,-2,0,2,4,6,8,10])
plt.show()

```
backpropagation 알고리즘은 미분가능한 함수를 activation함수로 사용해야하는데 그것이 가장알맞은 함수가 sigmoid함수입니다.<br>
출력 값을 1 또는 0만을 가질수 있습니다.<br>

![image](https://user-images.githubusercontent.com/29851990/147450092-3da6b918-7442-4309-8153-9b755a449980.png)

<br><br><br><br>

## Activation Function

### softmax
 - 출력계층에 많이 사용이 됩니다. 
 - 또한 앞서 말한대로 출력의 합계가 1인 activationFunc입니다. 
 - 또한 추가적으로 np.argmax()를 사용하여 가장 높은 값을 가진 요소의 인덱스를 결정할 수 있습니다.
 - linear는 다중출력을 할 수 있는 함수입니다. 하지만 역전파가 불가능합니다. 

<br>

### sigmoid
 - 출력 값을 0에서 1로 변경해줍니다. 
 - 하지만 밑의 함수처럼 입력신호의 총합이 크거나 작다면 gradient값이 0에 가까운 현상이 나타납니다.

![image](https://user-images.githubusercontent.com/29851990/147453673-c9b57886-17db-47f9-b2e8-2726bed8c634.png)

<br>

### tanh
 - 출력 값을 -1에서 1로 압축시키는 함수입니다. 
 - 하지만 이것또한 sigmoid와 마찬가지로 입력신호의 총합이 크거나 작다면 gradient값이 0에 가까운 현상이 나타납니다.

![image](https://user-images.githubusercontent.com/29851990/147453705-58a8b3de-0df7-492e-b542-b3851b2dbd29.png)

<br><br><br><br>

## Loss Function

### MSE
 - 평균제곱오차입니다. 
 - 예측 값이 실제 값으로부터 얼마나 떨어져있는지를 loss라고 하는데 MSE는 loss의 평균제곱오차를 구합니다. 
 - 특이점이 존재하면 수치가 많이 늘어나는 특징이 있습니다.

![image](https://user-images.githubusercontent.com/29851990/147453850-7668d58e-6f76-414e-b359-02c567bff6d1.png)

<br> 

### Categorical_Crossentropy  ||  Binary_Crossentropy
 - softmax를 사용할 때 사용하는 것이 좋은 선택입니다.

<br>

### MAE
 - MSE와 마찬가지로 loss를 구해 절대값으로 반환해 평균화시킨 것입니다. 
 - 그러므로 MSE와는 다르게 에러의 크기가 그대로 반영됩니다.

![image](https://user-images.githubusercontent.com/29851990/147453884-f843e605-4b39-41b7-9698-374815639f6e.png)

<br>

### 

<br><br><br><br>
## Optimizer

최적화를 통해 lossFunc을 최소화 시키는것입니다.<br>
손실이 특정수준으로 감소하면 모델이 입력을 출력에 매핑시킵니다.<br>

<br>

### metrics
 - 모델이 기본데이터 분포를 학습했는지 확인하는데 사용됩니다. 
 - accurate는 실제값에 기초한 정확한 예측의 백분율을 나타냅니다. 가장 많이 이용되는 것이 SGD, Adam, RMSprop입니다.

<br>

### Gradient Descent (GD)
 - 해당 함수의 최소값 위치를 찾기위해 비용합수의 그레디언트 반대 방향으로 정의한 step size를 가지고 조금씩 움직여가면서 최적의 파라미터를 찾으려는 방법입니다.
 - Batch Gradient Descent : 전체 데이터를 사용
 - Mini-batch Gradient Descent : 데이터의 일부만을 사용
 - Stochastic Gradient Descent : 하나의 데이터만을 사용

``` example GD

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

#임의로 x의 값을 100개 정합니다
#임의로 그래프를 하나 만듭니다
x = np.random.rand(100)
y = 0.5 * x + 0.2

#이따 weight값과 bias값이 갱신될때 설정한 주기마다 해당 그래프의 차이를 보여주는 함수를 만듭니다
def plot_prediction(y_pred, y) :
    plt.figure(figsize=(8,6))
    plt.scatter(x,y_pred)
    plt.scatter(x,y)
    plt.show()

#임의로 -1 에서 1사이의 값으로 weight와 bias값을 랜덤하게 설정합니다
W = np.random.uniform(-1,1)
b = np.random.uniform(-1,1)

#랜덤으로 설정된 weight 와 bias를 알맞은 값으로 변경될때까지 200번의 훈련을 시킵니다
for epoch in range(200) :

    #현재 설정된 weight 와 bias값으로 y의 예측값을 구합니다
    Y_pred = W * x + b

    #에러율이 0.001미만일 때 프로그램을 종료합니다
    error = np.abs(Y_pred - y).mean()
    if error < 0.001 :
        break

    #임의로 learning_rate를 설정합니다
    learning_rate = 0.7

    #해당 공식을 사용해 weight 와 bias를 Gradient Descent한 값을 구합니다
    W_gred = learning_rate * ((Y_pred - y)*x).mean()
    B_gred = learning_rate * (Y_pred - y).mean()

    #Gradient Descent한 weight 와 bias값을 갱신해줍니다
    W = W - W_gred
    b = b - B_gred

    #훈련 20번마다 한번씩 그래프를 띄워줍니다
    if epoch % 20 ==0 :
        Y_pred = W * x + b
        plot_prediction(Y_pred, y)


#cost함수 : 모델이 가정한 예측과 실제의 값이 얼마나 차이가 나는지를 나타내는 척도
#cost(w,b) = min시그마(Wxi + b + yi)^2
#cost함수를 쓸때 왜 최소제곱법을 쓰는가? 절댓값을 쓰게되면 컴퓨터 성능저하
#경사하강법(Gradient Descent) : cost함수를 줄이기위해 반복적으로 기울기를 계산하여 변수의값을 변경해나가는 과정
# y=x^2 기준)) 기울기를 조사해봤을때(미분) 음수라면 예측값이 실제값보다 왼쪽에 있으니까 오른쪽으로(+), 양수라면 왼쪽으로(-)
# 위의 x,y좌표계는 w,cost / b,cost 둘다 포함


```

Gradient Descent의 동작원리를 위의 코드로 나마 조금 더 쉽게 이해할 수 있습니다.


<br><br>

### SGD
 - StochasticGradientDescent의 약자로 GD를 변형시킨겁니다.
 - 먼저 GD란 모델이 가정한 예측과 실제의 값의 차이를 줄이기위해 반복적으로 기울기를 계산하여 특정변수를 변경해나가는 과정이고 SGD는 GD를 일부의 데이터만 사용하여 계산속도를 빠르게 만든 방법입니다.
 -  learning_rate는 보통 크게 0.1로 설정했다가 0.01처럼 점진적으로 줄여나가는 것이 좋습니다.

<br>

### Adam || RMSprop 
 - SGD의 learning_rate를 가진 변형이라고 볼수도 있습니다.

![image](https://user-images.githubusercontent.com/29851990/147450136-3b4e4d88-2054-4297-b0cd-72a5610c2c92.png)

<br><br><br><br>

## Overfit

### 과대적합을 방지하기 위한 전략
- 더 많은 훈련 데이터를 모읍니다.
- 네트워크의 용량을 줄입니다.
- 가중치 규제를 추가합니다.
- 드롭아웃을 추가합니다.

<br><br>

### 가중치 규제
 - 간단한 모델은 복잡한 것보다 과대적합되는 경향이 작을 것입니다.
 - 엔트로피가 작은 모델
 - 가중치가 작은 값을 가지도록 네트워크의 복잡도에 제약을 가하는 것
 - L1 규제 : 가중치의 절대값 합에 비례하여 가중치에 페널티를 주는 정규화 유형입니다. 희소 특성에 의존하는 모델에서 L1 정규화는 관련성이 없거나 매우 낮은 특성의 가중치를 정확히 0으로 유도하여 모델에서 해당 특성을 배제하는 데 도움이 됩니다. L2 정규화와 대비되는 개념입니다.
 - 희소특성 : 대부분의 값이 0이거나 비어 있는 특성 벡터입니다. 예를 들어 1 값 하나와 0 값 백만 개를 포함하는 벡터는 희소 벡터입니다. 또 다른 예로서, 검색어의 단어는 희소 특성일 수 있습니다. 특정 언어에서 가능한 단어는 무수히 많지만 특정 검색어에는 몇 개의 단어만 나오기 때문입니다.
 - L2 규제 : 가중치 제곱의 합에 비례하여 가중치에 페널티를 주는 정규화 유형입니다. L2 정규화는 높은 긍정 값 또는 낮은 부정 값(??)을 갖는 이상점 가중치를 0은 아니지만 0에 가깝게 유도하는 데 도움이 됩니다. L1 정규화와 대비되는 개념입니다. L2 정규화는 선형 모델의 일반화를 항상 개선합니다.
 - L1 규제는 일부 가중치 파라미터를 0으로 만듭니다. L2 규제는 가중치 파라미터를 제한하지만 완전히 0으로 만들지는 않습니다. 이것이 L2 규제를 더 많이 사용하는 이유 중 하나입니다.
 ``` overfit
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

<br>

### Drop out
 - 샘플들 보다 더 많은 특성(feature)들이 주어지면, 선형 모델은 오버핏(overfit) 될 수 있습니다.
 - 마지막 결과를 계산하기 위해서 사용되는 연산의 몇몇 뉴런들을 누락(drop out) 시킨다는 개념
 - 신경망 모델이 복잡해질 때 가중치 감소만으로는 어려운데 드롭아웃 기법은 뉴런의 연결을 임의로 삭제하는 것
 - 훈련할 때 임의의 뉴런을 골라 삭제하여 신호를 전달하지 않게 한다. 테스트할 때는 모든 뉴런을 사용한다.
 - ex) https://www.tensorflow.org/tutorials/keras/overfit_and_underfit?hl=ko

<br>

### Cross Entropy
 - 틀릴 수 있는 정보를 가지고 구한 최적의 엔트로피 값 즉, 불확실성 정보의 양
 - 틀릴 수 있는 정보의 예제로는 머신러닝 모델의 아웃풋(예측값)
 - 예측값을 통해 실제값과 얼마나 근사한지 알아보는 척도 
 - ![image](https://user-images.githubusercontent.com/29851990/147451292-96b0c3f8-df66-4c8d-a24a-9f34756ed7bb.png)
 - 예측값이 실제값과 완전히 동일할 때 엔트로피와 동일한 값이 나온다
 - 차이가 많이날 때 기하급수적으로 값이 증가한다.

<br><br><br><br>

## CNN

쓰이는 예시 
 - 이미지나 영상데이터를 처리할 때 쓰임

왜 쓰이게 되었나 
 - DNN의 문제점인 ovefiting에서 출발합니다. 
 - 너무 딮하게 훈련을하게되면 예를들어 강아지 사진같은 경우 내가 보여준 강아지사진이 아니고 조금만 다른 강아지사진이면 그것을 강아지라고 인식을 안하는 문제가 발생되어 적절히 특징을 잡아주는 방식이 필요했습니다. 
 - 그에 사용된 기술이 Convolutional 이고, 그 방식중에는 Zero Padding 과같은 기술이있습니다.
 - 그리고 Stride라는 개념이 있는데 이는 효율성을 나타냅니다. 
 - Convolutional을 할 때, 한칸한칸씩 움직이는 것이아니라 움직이는 거리를 정해주는것입니다.
 - 따라서, 이와같이 구조를 짜게되면 모든 레이어가 이어져있는 것이 아닌 필요없는 레이어들간의 관계에는 끊어나감으로서, overfiting을 막아냅니다.

![image](https://user-images.githubusercontent.com/29851990/147451468-257e091a-a59c-4cc9-8748-69246dab21ca.png)

<br>

### Regularization

일반적으로 사용하는 정규화의 방법에는 Dropout이 있습니다.<br>
사용방법은 Dropout(dropoutIndex = 0.45)와 같이 값을 넣어주면 다음 layer에 참여할 노드의 일부를 무작위로 제거하는 방법입니다.<br>
예를들어서 첫 번째 layer의 노드가 256이라면 (1-0.45) * 256을 하여 다음 layer에 전달하는 방식입니다.<br>
이것은 안정적인 뉴런네트워크를 만듭니다. <br>
다만 dropout은 출력계층에서 사용하지 않고, 또한 훈련 중에만 활성화 해야하는 주의해야할 점이 있습니다.<br>
또한, 과대적합을 방지하기 위해 사용되는 방법은 L1 규제와 L2규제가 있습니다. <br>
여기서 L1규제는 파라미터값을 0으로 만들기 때문에 일반적으로는 L2규제를 사용합니다. <br>
L2규제란 가중치 제곱의 합에 비례하여 가중치에 패널티를 주는 정규화유형입니다. <br>
L2 정규화는 가중치를 0은 아니지만 0에 가깝게 유도하는데 도움을 줍니다.

<br>

### Convolution
MLP에서 Dense Layer가 노드의 수를 정한다면, CNN에서는 kernel이 그 작업을 합니다. <br>
또한 Convolution이란 합성 곱 처리 결과로부터 Feature Map을 만드는 연산을 칭합니다.<br>
다만, 입력과 출력의 Feature Map 치수가 동일해야하는 경우 option='same'을 사용합니다.<br>
그러면 입력에서 Convolution 후에 치수를 변경하지 않도록 경계 주위에 0으로 채우게됩니다.<br>
이러한 방법을 zero padding이라고 합니다.

<br><br>

### Pooling operations

CNN을 사용하면서 추가한 Layer중 Pooling Layer가 있습니다.<br>
MaxPooling2D는 아래의 그림과 같이 patch_size를 1로 줄이고 그 영역의 Max값을 구해 새로운 Feature Map를 만듭니다.<br>

![image](https://user-images.githubusercontent.com/29851990/147455079-77b0083c-833b-492c-a5a1-b2ad780e75d6.png)

MaxPooling2D의 큰 특징은 map_size를 줄이는데에 있습니다.<br>
또한 Pooling 방식에는 Average와 Min도 있습니다.
