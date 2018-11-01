from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
		include_top=False,
		input_shape=(150, 150, 3))

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

train_dir='/home/guohongliang/project/keras/dog_and_cat/train_dir'
val_dir  ='/home/guohongliang/project/keras/dog_and_cat/val_dir'
test_dir ='/home/guohongliang/project/keras/dog_and_cat/test_dir'

import os
import numpy as np 

def extract_fetures(directory, sample_count):
	features = np.zeros(shape=(sample_count, 4, 4, 512))
	labels   = np.zeros(shape=(sample_count))

	batch_size = 20
	generator = datagen.flow_from_directory(
			directory,
			target_size=(150, 150),
			batch_size=batch_size,
			class_mode='binary')

	i = 0
	for inputs_batch, labels_batch in generator:
		features_batch = conv_base.predict(inputs_batch)
		features[i * batch_size : (i + 1) * batch_size] = features_batch
		labels[i * batch_size : (i + 1) * batch_size]   = labels_batch
		i += 1
		if i * batch_size >= sample_count:
			break
	
	return features, labels

train_features, train_labels = extract_fetures(train_dir, 2000)
val_features,   val_labels   = extract_fetures(val_dir,   1000)
test_features,  test_labels  = extract_fetures(test_dir,  1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
val_features   = np.reshape(val_features,   (1000, 4 * 4 * 512))
test_features  = np.reshape(test_features,  (1000, 4 * 4 * 512))

from keras import layers
from keras import models
from keras import optimizers
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
		optimizer=optimizers.RMSprop(lr=2e-5),
		metrics=['acc'])

train_hist = model.fit(train_features, train_labels,
		epochs=30,
		batch_size=20,
		validation_data=(val_features, val_labels))

import matplotlib.pyplot as plt

loss     = train_hist.history['loss']
val_loss = train_hist.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('ld3.1.png')

plt.clf()
acc     = train_hist.history['acc']
val_acc = train_hist.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('ad3.1.png')
