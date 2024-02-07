

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

data = open('misc/input.txt').read()

corpus = data.lower().split('\n')

print(corpus[0])


#instantiate the tokenizer
tokenizer = Tokenizer()

#generate the word index dictionary
tokenizer.fit_on_texts(corpus)

#define the total words. add 1 to include index 0 wich is just the padding token
total_words = len(tokenizer.word_index) + 1

print(f'word index dictionary: {tokenizer.word_index}')
print(f'total words: {total_words}')


#preprocessing the dataset

input_sequences = []

for line in corpus:
    #tokenize the current line
    token_list = tokenizer.texts_to_sequences([line])[0]
    print(token_list)

    #loop over the line several times to generate the subphrases
    for i in range(1, len(token_list)):
        #generate the subphrase
        n_gram_sequence = token_list[:i+1]
        #appen subphrase to the sequence list
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

#padd all sequences
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_len, padding = 'pre')

#create inputs and label by splitting the last token in the subphrase
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes = total_words)


#sanity checks
sentence = corpus[0].split()
print(f'sample sentence: {sentence}')
token_list = []
for word in sentence:
    token_list.append(tokenizer.word_index[word])
print(token_list)

elem_num = 5

print(f'token list: {xs[elem_num]}')
print(f'decoded to text: {tokenizer.sequences_to_texts([xs[elem_num]])}')

print(f'one hot label: {ys[elem_num]}')
print(f'index of label: {np.argmax(ys[elem_num])}')


elem_num = 4

print(f'token list: {xs[elem_num]}')
print(f'decoded to text: {tokenizer.sequences_to_texts([xs[elem_num]])}')

print(f'one hot label: {ys[elem_num]}')
print(f'index of label: {np.argmax(ys[elem_num])}')



#build and compile the model
embedding_dim = 100
lstm_units = 150
lr = 0.01

model = Sequential([
    Embedding(total_words, embedding_dim, input_length = max_sequence_len-1),
    Bidirectional(LSTM(lstm_units)),
    Dense(total_words, activation = 'softmax')
])

model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
              metrics = ['accuracy'])

model.summary()


history = model.fit(xs, ys, epochs = 100)

import matplotlib.pyplot as plt

# Plot utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

# Visualize the accuracy
plot_graphs(history, 'accuracy')



#generating text
# Define seed text
seed_text = "You spin me right round baby right round"

# Define total words to predict
next_words = 100

# Loop until desired length is reached
for _ in range(next_words):

    # Convert the seed text to a token sequence
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    # Feed to the model and get the probabilities for each index
    probabilities = model.predict(token_list, verbose=0)

    # Get the index with the highest probability
    predicted = np.argmax(probabilities, axis=-1)[0]

    # Ignore if index is 0 because that is just the padding.
    if predicted != 0:
        # Look up the word associated with the index.
        output_word = tokenizer.index_word[predicted]

        # Combine with the seed text
        seed_text += " " + output_word

# Print the result
print(seed_text)