import keras.callbacks
import tensorflow as tf
# import pandas as pd
# import sklearn
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

# fashion mnist dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print(len(train_images), len(train_labels))
print(len(test_images), len(test_labels))

# normalize pixel values
train_images = train_images / 255.
test_images = test_images / 255.

# print label, pixel values, and show an image of the object
#print(f'label: {train_labels[0]}')
#print(f'pixel array: \n {train_images[0]}')
#plt.imshow(train_images[0], cmap='Greys')
#plt.show()
'''
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
preds = model.predict(test_images)
print(preds[0])
print(np.argmax(preds, axis=1))


#fitting on 512 neurons
model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['accuracy'])

model2.fit(train_images, train_labels, epochs=5)
model2.evaluate(test_images, test_labels)


#fitting on 1024 neurons
model3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['accuracy'])

model3.fit(train_images, train_labels, epochs=5)
model3.evaluate(test_images, test_labels)

'''

#callback to stop when accuracy hits above 60%
class myCallback(tf.keras.callbacks.Callback):
  '''
  Halts the training when the loss falls below 0.4

  Args:
    epoch (integer) - index of epoch (required but unused in the function definition below)
    logs (dict) - metric results from the training epoch
  '''
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss') < 0.4): # Experiment with changing this value
      print("\nLoss is low, cancelling training")
      self.model.stop_training = True


callbacks = myCallback() #instantiate the callbacks class

model4 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model4.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])