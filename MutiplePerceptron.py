import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#inputData01에서 0과 1을 적절히 조합할만한 배열을 만듭니다.
inputData01 = np.array([0,0,1,1,0,1,0,1]).reshape(2,4)

#inputData02에서 0과 1을 적절히 조합할만한 배열을 만듭니다.
inputData02 = np.array([1,1,-1,-1,1,-1,1,-1]).reshape(2,4)

#퍼셉트론이란 다수의 입력을 받아 특정알고리즘을 바탕으로 하나의 신호를 출력하여 그결과에 따라 흐른다(1), 안흐른다(0)으로 출력하는 알고리즘을 말합니다.
#단층퍼셉트론으로는 XOR게이트를 만들 수 없으므로 다중퍼셉트론을 구현합니다.
#0,0 / 1,1 일때와 1,0 / 0,1일때로 나눈 결과값을 만들수 있는 함수를 만들어줍니다.
#perceptron1,2,3은 inputData01에 해당하는 perceptron
#perceptron4,5,6은 inputData02에 해당하는 perceptron

# (0,0)이 아니라면 무조건 1반환
def perceptron1 (input1, input2) :
    weight1 = 1
    weight2 = 1
    bias = 0
    result = (weight1 * input1) + (weight2 * input2) + bias;

    if(result > 0) :
        return 1
    else :
        return 0

# (1,1)이 아니라면 무조건 0반환
def perceptron2(input1, input2):
    weight1 = 1
    weight2 = 1
    bias = -1
    result = (weight1 * input1) + (weight2 * input2) + bias;

    if (result > 0):
        return 1
    else:
        return 0

#같으면 0, 다르면 1이되게 함수구현 다를때는 무조건 1,0이 perceptron3함수에 인자로 전달되게끔 perceptron1,2를 설계
def perceptron3(input1, input2):
    weight1 = 1
    weight2 = -2
    bias = 0
    result = (weight1 * input1) + (weight2 * input2) + bias;

    if (result > 0):
        return 1
    else:
        return 0

# (-1,-1)이아니라면 무조건 1반환
def perceptron4 (input1, input2) :
    weight1 = 1
    weight2 = 1
    bias = 2
    result = (weight1 * input1) + (weight2 * input2) + bias;

    if(result > 0) :
        return 1
    else :
        return 0

# (1,1) 이아니라면 무조건 0반환
def perceptron5(input1, input2):
    weight1 = 1
    weight2 = 1
    bias = -1
    result = (weight1 * input1) + (weight2 * input2) + bias;

    if (result > 0):
        return 1
    else:
        return 0

#같으면 0, 다르면 1이되게 함수구현 다를때는 무조건 1,0이 perceptron6함수에 인자로 전달되게끔 perceptron4,5를 설계
def perceptron6(input1, input2):
    weight1 = 1
    weight2 = -2
    bias = 0
    result = (weight1 * input1) + (weight2 * input2) + bias;

    if (result > 0):
        return 1
    else:
        return 0

for i in range(0,4) :
    print("입력 :" + str(inputData01[0,i]) + "," + str(inputData01[1,i]) + " 출력 :" + str(perceptron3(perceptron1(inputData01[0,i],inputData01[1,i]),perceptron2(inputData01[0,i],inputData01[1,i]))))

for i in range(0,4) :
    print("입력 :" + str(inputData02[0,i]) + "," + str(inputData02[1,i]) + " 출력 :" + str(perceptron6(perceptron4(inputData02[0,i],inputData02[1,i]),perceptron5(inputData02[0,i],inputData02[1,i]))))

