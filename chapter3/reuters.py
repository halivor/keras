from keras.datasets import reuters

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
KTF.set_session(session )

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

import numpy as np
def vs(seq, dim=10000):
	res = np.zeros((len(seq), dim))
	for i, p in enumerate(seq):
		res[i, p] = 1.
	return res

x_train = vs(train_data)
x_test  = vs(test_data)

def to_one_hot(labels, dim=46):
	res = np.zeros((len(labels), dim))
	for i, l in enumerate(labels):
		res[i, l] = 1.
	return res

x_train_labels = to_one_hot(train_labels)
x_test_labels  = to_one_hot(test_labels)
# x_train_labels = np.array(train_labels)
# x_test_labels  = np.array(test_labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
		loss='categorical_crossentropy',
		metrics=['accuracy'])
# model.compile(optimizer='rmsprop',
		# loss='sparse_categorical_crossentropy',
		# metrics=['accuracy'])

x_val    = x_train[:500]
xp_train = x_train[500:]
y_val    = x_train_labels[:500]
yp_train = x_train_labels[500:]

history = model.fit(xp_train, yp_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))

# np.set_printoptions(threshold=np.inf)
v = model.evaluate(x_test, x_test_labels)
print(v)
p = model.predict(x_test)
print(p)

