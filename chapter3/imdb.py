from keras.datasets import imdb


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 将索引映射为单词
# word_index = imdb.get_word_index()
# print(word_index)
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# print(reverse_word_index)
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(train_data[0])
# print(decoded_review)

import numpy as np
def vs(seq, dim=10000):
	res = np.zeros((len(seq), dim))
	for i, p in enumerate(seq):
		res[i, p] = 1.
	return res


x_train = vs(train_data)
x_test  = vs(test_data)

# np.set_printoptions(threshold=np.inf)

y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')


from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1,  activation='sigmoid'))
# model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000, )))
# model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
# model.add(layers.Dense(1,  activation='sigmoid'))

x_val    = x_train[:5000]
xp_train = x_train[5000:]

y_val    = y_train[:5000]
yp_train = y_train[5000:]

print(len(x_val), len(xp_train))

model.compile(optimizer='rmsprop',
		loss='binary_crossentropy',
		metrics=['accuracy'])

# model.fit(xp_train, yp_train, epochs = 4, batch_size = 1024)
# pr = model.predict(x_test)
# print(pr)

history = model.fit(xp_train, yp_train, epochs = 10, batch_size = 512, validation_data=(x_val, y_val))

pr = model.predict(x_test)
print(pr)
results = model.evaluate(x_val, y_val)
print(results)

import matplotlib.pyplot as plt

history_dict    = history.history
loss_values     = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ld.png')

plt.clf()
acc     = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('ad.png')

