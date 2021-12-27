import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train) ,(x_test, y_test) = mnist.load_data() #x가 문제 y가 정답같은 느낌
unique, counts = np.unique(y_train, return_counts=True)#np.unique(return_count=True)란 지정된 배열안의 요소의 중복을 피해서의 출력과 그 요소의 개수를 배열로 출력하는 것
print("Train labels : ", dict(zip(unique,counts)))

unique, counts = np.unique(y_test,return_counts=True)
print("Test labels: ",dict(zip(unique,counts)))

indexes = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5,5))
for i in range(len(indexes)) :
    plt.subplot(5,5,i+1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.savefig("mnist-samples.png")#C:\Users\cha\PycharmProjects\pythonProject\venv 안에저장
plt.show()
plt.close()
