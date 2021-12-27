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