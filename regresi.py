import tensorflow as tf
import numpy as np
from tensorflow import keras

xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
ys = np.array([4, 6, 8, 10, 12, 14], dtype=float)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1]),
keras.layers.Dense(units=5),
keras.layers.Dense(units=1)])
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=50)

print(model.predict([10]))