import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


train_dir = '../../images/rps/train'
test_dir = '../../images/rps/test'
valid_dir = '../../images/rps/validation'

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=20,
                                   horizontal_flip=True,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    batch_size=128,
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255.)
validation_generator = validation_datagen.flow_from_directory(test_dir,
                                                              batch_size=128,
                                                              target_size=(150,150),
                                                              class_mode='categorical')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation = 'softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss= tf.keras.losses.CategoricalCrossentropy(),
              metrics = ['accuracy'])

history = model.fit(train_generator,
                    epochs=25,
                    steps_per_epoch=20,
                    validation_data = validation_generator,
                    verbose = 1,
                    validation_steps=3)


