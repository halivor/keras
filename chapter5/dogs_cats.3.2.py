from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
		include_top=False,
		input_shape=(150, 150, 3))

from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
conv_base.trainable=False
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest'
		)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_dir='/home/guohongliang/project/keras/dog_and_cat/train_dir'
train_generator = train_datagen.flow_from_directory(
		train_dir,
		target_size=(150, 150),
		batch_size=20,
		class_mode='binary')

validation_dir='/home/guohongliang/project/keras/dog_and_cat/val_dir'
validation_generator = test_datagen.flow_from_directory(
		validation_dir, 
		target_size=(150, 150),
		batch_size=20,
		class_mode='binary')

model.compile(
		optimizer=optimizers.RMSprop(lr=2e-5),
		loss='binary_crossentropy',
		metrics=['acc'])

train_hist = model.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=30,
		validation_data=validation_generator,
		validation_steps=50)

import matplotlib.pyplot as plt

loss     = train_hist.history['loss']
val_loss = train_hist.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('ld3.2.png')

plt.clf()
acc     = train_hist.history['acc']
val_acc = train_hist.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('ad3.2.png')
