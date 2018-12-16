# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:16:58 2018

model.py

@author: Admin
"""

import numpy as np
import glob

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing import image


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

model = Sequential()
# Layer 1
model.add(Conv2D(32, (5, 5), input_shape=(150, 150, 3)))
#              #filter,filter,input_shape
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 5
#model.add(Conv2D(120, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(190))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 64

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

#this is a generator that will read pictures found in
#subfolers of 'data/train', and indefinitely generate
#batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'dataset/Train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'dataset/Test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

#%% Train
model.fit_generator(
        train_generator,
        steps_per_epoch=100 // batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=10 // batch_size)
model.save_weights('100EPOCH.h5')  

#%% Test
model.load_weights('100EPOCH.h5')

# for showing test result from test dataset
def show_test(folder_dir):
    count=0
    for filename in glob.glob(folder_dir + '*.png'):
            test_image = image.load_img(folder_dir + filename[len(folder_dir):], 
                    target_size = (150, 150))
            plt.imshow(test_image)
            plt.show()
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            train_generator.class_indices
            if result[0][0] == 1:
                prediction = 'exam'
                print(prediction)
            else:
                count+=1
                prediction = 'lecture'
                print(prediction)


#folder_dir = 'dataset/Test/exam/'
folder_dir = 'dataset/Test/lecture/'
show_test(folder_dir)












