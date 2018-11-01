from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')

model.summary()
exit(0)

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

from keras import optimizers
model.compile(loss='binary_crossentropy',
		optimizer=optimizers.RMSprop(lr=1e-4),
		metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
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

train_hist = model.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=30,
		validation_data=validation_generator,
		validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

import matplotlib.pyplot as plt

loss     = train_hist.history['loss']
val_loss = train_hist.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('ld1.png')

plt.clf()
acc     = train_hist.history['acc']
val_acc = train_hist.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('ad1.png')
