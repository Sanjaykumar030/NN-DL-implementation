import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

np.random.seed(1)
tf.random.set_seed(1)

X = np.random.randn(500, 2).astype(np.float32)
Y = (np.sum(X, axis=1) > 0).astype(np.float32)

model = models.Sequential([
    layers.Dense(4, activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01), bias_initializer=tf.keras.initializers.Zeros(), input_shape=(2,)),
    layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01), bias_initializer=tf.keras.initializers.Zeros())
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, Y, epochs=100, batch_size=30, verbose=1)

loss, acc = model.evaluate(X, Y, verbose=0)
print(f"Training accuracy: {acc * 100:.2f}%")
