import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150,150,3),
                                include_top = False,
                                weights = None)


pre_trained_model.load_weights(local_weights_file)

#locking pretrained model layers
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

#the bottom layers have convolved to 3x3 but we want to use something with a little more information
#so we set the last layer to one with a 7x7 convolution (mixed7)
last_layer = pre_trained_model.get_layer('mixed7')

last_output = last_layer.output


x = layers.Flatten()(last_output)

x = layers.Dense(1024, activation = 'relu')(x)

#add dropout layer to reduce overfitting dropping 20% of neurons
x = layers.Dropout(0.2)(x)

x = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale = 1./255.)



train_dir = '../../images/cats_and_dogs_filtered/train/'
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 20,
    class_mode = 'binary',
    target_size = (150,150)
)

validation_dir = '../../images/cats_and_dogs_filtered/validation'
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    batch_size = 20,
    class_mode = 'binary',
    target_size = (150,150)
)


history = model.fit(train_generator,
                    validation_data = validation_generator,
                    steps_per_epoch= 100,
                    epochs = 50,
                    validation_steps= 50)



import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()