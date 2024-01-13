from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf





train_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (300,300),
                                                    batch_size = 128,
                                                    class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255.)

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (300,300),
                                                        batch_size = 32,
                                                        class_mode = 'binary')


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (300,300,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(16, (2,2), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(16, (2,2), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics = ['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch = 8,  #there are 1024 images in the training directory so we're loading them in 128 at a time
                    epochs = 15,
                    validation_data = validation_generator,
                    validation_steps = 8,
                    verbose = 2)


