from keras.datasets import imdb
from keras          import preprocessing
max_fetures = 10000
maxlen=20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_fetures)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen)
x_test  = preprocessing.sequence.pad_sequences(x_test , maxlen = maxlen)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

hist = model.fit(x_train, y_train,
				 epochs=10,
				 batch_size=32,
				 validation_split=0.2)

import numpy as np
np.set_printoptions(threshold=np.inf)
print(hist.history)

