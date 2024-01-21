import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['I love my dog', 
'I love my cat', 
'You love my dog!',
'Do you think my dog is amazing?']

#tokenizer strips punctation and makes everything lower case
tokenizer = Tokenizer(num_words= 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
padded = pad_sequences(sequences, padding = 'post', truncating = 'post', maxlen = 5)
print(padded)

test_data = ['I really love my dog', 'my dog really loves my manatee']

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)