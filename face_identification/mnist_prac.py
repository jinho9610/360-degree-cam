
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('mnist_dnn_model.h5')

new_model.summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print('shape of x_train:', x_train.shape)
print('shape of y_train:', y_train.shape)
print('shape of x_test:', x_test.shape)
print('shape of y_test:', y_test.shape)

# plt.rcParams['figure.figsize'] = (5, 5)
# plt.imshow(x_train[0])
# plt.show()

# reshape and normalization

x_train = x_train.reshape((60000, 28 * 28)) / 255.0
x_test = x_test.reshape((10000, 28 * 28)) / 255.0

# Sequential model

model = tf.keras.models.Sequential()

# Stacking layers
model.add(tf.keras.layers.Dense(
    units=128, activation='relu', input_shape=(28*28,)))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary()

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(x_train, y_train, epochs=5, verbose=1, validation_split=0.2)

model.evaluate(x_test, y_test)

preds = model.predict(x_test, batch_size=128)

print(preds[0])

np.argmax(preds[0])


print(y_test[0])
plt.imshow(x_test[0].reshape(28, 28))
plt.show()

model.save('mnist_dnn_model.h5')
