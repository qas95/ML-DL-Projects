# -*- coding: utf-8 -*-
"""Classifying Traffic Signs using CNNs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gbh7NMo6QDSXxGZUU1t3fc5XKL4QbmuW
"""

!git clone https://bitbucket.org/jadslim/german-traffic-signs

!ls german-traffic-signs/

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

import random
import pickle
import pandas as pd
import cv2
import requests
from PIL import Image

with open('german-traffic-signs/train.p', 'rb') as f:
  train_data = pickle.load(f)

with open('german-traffic-signs/test.p', 'rb') as f:
  test_data = pickle.load(f)

with open('german-traffic-signs/valid.p', 'rb') as f:
  valid_data = pickle.load(f)

X_train = train_data['features']
y_train = train_data['labels']

X_test = test_data['features']
y_test = test_data['labels']

X_val = valid_data['features']
y_val = valid_data['labels']

assert(X_train.shape[0] == y_train.shape[0])
assert(X_test.shape[0] == y_test.shape[0])
assert(X_val.shape[0] == y_val.shape[0])

data = pd.read_csv('german-traffic-signs/signnames.csv')
data.head()

num_of_samples=[]

cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
fig.tight_layout()

for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
          axs[j][i].set_title(str(j) + " - " + row["SignName"])
          num_of_samples.append(len(x_selected))
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

plt.imshow(X_train[100])
plt.show()

print(data.iloc[y_train[100]]['SignName'])

def grayScale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, )
  return img

plt.imshow(grayScale(X_train[1]), cmap = plt.get_cmap('gray'))

def equalize_histogram(img):
  img = cv2.equalizeHist(img)
  return img

def preprocessing(img):
  img = grayScale(img)
  img = equalize_histogram(img)
  img = img/255
  return img

#Preprocessing all the images at once

X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_val = np.array(list(map(preprocessing, X_val)))

plt.imshow(X_train[100], cmap = 'gray')

X_train.shape
X_test.shape
X_val.shape

#Reshaping (adding depth) the data to feed into the neural net

X_train = X_train.reshape(34799, 32, 32,1)
X_test = X_test.reshape(12630, 32, 32,1)
X_val = X_val.reshape(4410, 32, 32,1)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)

y_train[1]

plt.imshow(X_train[1].reshape(32,32))

def leNet():
  model = Sequential()
  
  model.add(Conv2D(8, (5, 5), padding="same", activation='relu', input_shape=(32,32,1)))
  model.add(BatchNormalization(axis=-1))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(16, (3,3), activation='relu'))
  model.add(BatchNormalization(axis=-1))
  model.add(Conv2D(16, (3,3), activation='relu'))
  model.add(BatchNormalization(axis=-1))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(32, (3,3), activation='relu'))
  model.add(BatchNormalization(axis=-1))
  model.add(Conv2D(32, (3,3), activation='relu'))
  model.add(BatchNormalization(axis=-1))
  model.add(MaxPooling2D(2,2))
  
  model.add(Flatten())
  model.add(Dense(units = 200, activation = 'relu'))
  model.add(BatchNormalization(axis=-1))
  model.add(Dropout(0.5))
  
  model.add(Flatten())
  model.add(Dense(units = 200, activation = 'relu'))
  model.add(BatchNormalization(axis=-1))
  model.add(Dropout(0.5))
  
  model.add(Dense(43, activation = 'softmax'))
  model.compile(optimizer = Adam(lr = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model

model = leNet()
model.summary()

X_train.shape

#hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 10, batch_size = 4, shuffle=True, verbose = 1)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['Training', "validation"])
plt.title('Accuracies')
plt.xlabel('epochs')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Training', "validation"])
plt.title('Loss')
plt.xlabel('epochs')
plt.show()

score = model.evaluate(X_test, y_test)
print('Loss ',score[0])
print('Accuracy ',score[1])

url = 'https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg'
response = requests.get(url, stream = True)
img2 = Image.open(response.raw)

plt.imshow(img2)

img2 = np.asarray(img2)
img2 = cv2.resize(img2, (32,32))
img2.shape

img2 = grayScale(img2)
img2 = equalize_histogram(img2)
plt.imshow(img2)
img2.shape

np.argmax(model.predict(img2.reshape(1,32,32,1)), axis = -1)
#Incorrect Classification

data.iloc[38]

#Since there is not enough samples for each of the 43 classes, lets use data augmentation and create more training data

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,shear_range=0.1, zoom_range=0.2, rotation_range=10)
datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20)
x, y = next(batches)

#Displaying a random batch of 20 newly generated images

plt.figure(figsize = (10,8))
fig, axis = plt.subplots(5,4)
l = len(x)
a=0
for i in range(4):
  for j in range(5):
    if(a < l):
      axis[j][i].imshow(x[a].reshape(32,32), cmap = 'gray')
      a+=1

model.fit_generator(datagen.flow(X_train, y_train, batch_size=5), steps_per_epoch=len(X_train)/5, epochs = 10, verbose = 1, validation_data=(X_val, y_val))

#Lets try the same image again after training the model with augmented data
plt.imshow(img2)

np.argmax(model.predict(img2.reshape(1,32,32,1)), axis = -1)

data.iloc[22]

plt.imshow(img2.reshape(32,32))

