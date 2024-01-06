import tensorflow as tf
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(pd.__version__)
print(sklearn.__version__)
print(matplotlib.__version__)
print(sns.__version__)


# The Hello World of nns
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.arange(0,7, dtype=float)
y = np.arange(.5, 4.0, .5, dtype=float)
print(x)
print(y)

model.fit(x, y, epochs=500)


print(model.predict([7]))
print(model.predict([10.0]))