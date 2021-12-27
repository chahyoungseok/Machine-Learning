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