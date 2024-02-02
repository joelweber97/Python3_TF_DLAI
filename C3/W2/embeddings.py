import tensorflow as tf
print(tf.__version__)
import tensorflow_datasets as tfds
import numpy as np

imdb, info = tfds.load('imdb_reviews', with_info = True, as_supervised= True)
train_data, test_data = imdb['train'], imdb['test']


training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s,l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

for s,l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

# expect labels to be in np arrays
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)


vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_token = '<OOV>'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token= oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen = max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length)


#define nn
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Flatten(),  #can either use flatten
    #tf.keras.layers.GlobalMaxPool1D(), #or a gap 1d which includes some averaging to flatten it out
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

model.summary()

model.fit(padded, training_labels_final, epochs = 10, steps_per_epoch= 32, validation_data = (testing_padded, testing_labels_final))

#32/32 [==============================] - 1s 47ms/step - loss: 0.0764 - accuracy: 0.9887 - val_loss: 0.3899 - val_accuracy: 0.8374
#looks like the current method is overfitting. we'll look at strategies to reduce overfitting shortly.


e = model.layers[0]  #getting the results for the embedding layer (layer 0)
weights = e.get_weights()[0]
print(weights.shape) #shape is (vocab_size, embedding_dim)
#returns  (10000, 16)

'''
#to plot it we need to reverse our word index
Hello:1
World:2
How:3
Are:4
You:5

'''
reverse_word_index = tokenizer.index_word

'''
1: Hello
2: World
3: How
4: Are
5: You
'''

#write vectors and metadata out to files

import io
out_v = io.open('vecs.tsv', 'w', encoding = 'utf-8')
out_m = io.open('meta.tsv', 'w', encoding = 'utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()

#we can to go tensorflow projector website and load the data in there to see the project of the data.








