import tensorflow as tf
import numpy as np

# Model Definition

class TwoLayerNN(tf.keras.Model):
  def __init__(self, n_x, n_h, n_y):
    super(TwoLayerNN, self).__init__()
    self.hidden = tf.keras.layers.Dense(
        n_h, activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),bias_initializer=tf.keras.initializers.Zeros()
 )
    self.output_layer = tf.keras.layers.Dense(
        n_y, activation = 'sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),bias_initializer=tf.keras.initializers.Zeros()  

  )
  def call(self, X):
    A1 = self.hidden(X)
    A2 = self.output_layer(A1)
    return A2
if __name__ == '__main__':
  np.random.seed(1)
  tf.random.set_seed(1)


  # Dummy data
  X = np.random.randn(2, 500).astype(np.float32)
  Y = (np.sum(X, axis=0) > 0).astype(np.float32).reshape(-1,1)

  X_tf = tf.convert_to_tensor(X.T)
  Y_tf = tf.convert_to_tensor(Y)

  model = TwoLayerNN(n_x=2, n_h=4, n_y=1)
  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
  model.fit(X_tf, Y_tf, epochs=500, verbose=0)
  loss, acc = model.evaluate(X_tf, Y_tf, verbose=0)
  print(f"Training accuracy: {acc * 100:.2f}%")
  
