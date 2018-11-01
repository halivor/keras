import os

imdb_dir  = '/home/guohongliang/project/keras/chapter6/aclImdb'
test_dir = os.path.join(imdb_dir, 'train')

labels = []
texts  = []

for label_type in ['neg', 'pos']:
	dir_name = os.path.join(test_dir, label_type)
	for fname in os.listdir(dir_name):
		if fname[-4:] == '.txt':
			f = open(os.path.join(dir_name, fname))
			texts.append(f.read())
			f.close()
			if label_type == 'neg':
				labels.append(0)
			else:
				labels.append(1)

from keras.preprocessing.text     import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100
training_samples = 200
val_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen = maxlen)

labels = np.asarray(labels)
print('shape of data tensor:' , data.shape)
print('shape of label tensor:', labels.shape)

embedding_dim = 100

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.load_weights('pre_trained_glove_model.h5')
res = model.evaluate(data, labels)
print(res)

