import tensorflow as tf 
import cv2, os, glob
import numpy as np 

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

from skimage import io
from sklearn.model_selection import train_test_split

from PIL import Image

def read_data_from_dir(train_dir, test_dir):
    train_data = []
    train_label = []

    test_data = []
    test_label = []

    # Add all the training images and labels to their proper arrays
    for img in train_dir:
        current_img = io.imread(img, as_gray=True) # Read in image as grayscale

        # Do edge extraction here if necessary
        current_img = cv2.Canny(current_img, 100, 200)

        train_data.append(current_img)
        train_label.append(img[-6]) # The class label for each image is 6th to last character in the file name

    # Add all the test images and labels to their proper arrays
    for img in test_dir:
        current_img = io.imread(img, as_gray=True) # Read in image as grayscale

        # Do edge extraction here if necessary
        current_img = cv2.Canny(current_img, 100, 200)

        test_data.append(current_img)
        test_label.append(img[-6]) # The class label for each image is 6th to last character in the file name

    # Return as 4 numpy arrays
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

# File paths of training and testing images
# Uses glob to grab all .png files in the directory
train_img_dir = glob.glob("../images/fingers/train/*.png")
test_img_dir = glob.glob("../images/fingers/test/*.png")

# Get and display number and training and testing images
num_train_imgs = len(train_img_dir)
num_test_imgs = len(test_img_dir)

print("There are {} training images and {} testing images".format(num_train_imgs, num_test_imgs))

# Get the data and labels
train_imgs, train_labels, test_imgs, test_labels = read_data_from_dir(train_img_dir, test_img_dir)

# Reshape the data into the proper shape
train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2], 1)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=6)

test_imgs = test_imgs.reshape(test_imgs.shape[0], test_imgs.shape[1], test_imgs.shape[2], 1)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=6)

print(train_imgs.shape, train_labels.shape, test_imgs.shape, test_labels.shape)

train_imgs = train_imgs/255


# Define our CNN
model = Sequential()

# Takes in a 128x128 grayscale image
model.add(Conv2D(32, (3,3), input_shape = (128, 128, 1), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(128, (3,3), activation = 'relu'))

model.add(Flatten())

model.add(Dropout(0.40))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.40))
# 6 node output for the 6 different classes 0-5 fingers
# Softmax activation provides a 0-1 activation value which corresponds to 
# % that an image matches with a defined class
model.add(Dense(6, activation = 'softmax'))

# Show an easier to read summary of the structure of the model
model.summary()

# Split into training and validation data
img_train, img_test, label_train, label_test = train_test_split(train_imgs, train_labels, test_size = 0.20, random_state = 7, shuffle = True)
img_train_val, img_test_val, label_train_val, label_test_val = train_test_split(img_train, label_train, test_size = 0.20, random_state = 7, shuffle = True)

# Compile the model with the optimizer, loss function, and desired metric
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with our data 
model.fit(x = img_train, y = label_train, batch_size = 128, epochs = 10, validation_data=(img_test, label_test))

model.save("../models/edges_model.h5")
