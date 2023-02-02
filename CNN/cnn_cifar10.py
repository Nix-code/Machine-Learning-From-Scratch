
# image classification on cifar10 dataset

from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import  matplotlib.pyplot as plt
# load the dataset
class_names = ['airplanes','cars','birds','cats','deer','dogs','frogs','horses','ships','trucks']
(train_images, train_labels),(test_image, test_labels) = datasets.cifar10.load_data();



model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))

import tensorflow as tf
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

hist = model.fit(train_images, train_labels, epochs=10, validation_data=(test_image, test_labels))


