"""
Author: Ryan O'Shea

Description: 
Class definition and associated testing code for neural network 
class. Cabable of training a model, loading a model, and evaluating a model. The
Test code in the main function should test all the functions in the class to make 
the model and class are working as intended.

********** IMPORTANT **********
Training the model is very computationally demanding and should only be done on a 
computer with a relatively powerful GPU and a fair amount of memory.

I trained all my models on a computer with the following specs:
CPU: Ryzen 5 1600
GPU: GTX 1070
RAM: 16GB DDR4

I also attempted to train on my laptop that had a weak CPU, no GPU, and only 8GB
of RAM and it was unable to run because tensorflow could not allocate enough memory.
********** IMPORTANT **********

If you need to change where models are saved to and loaded changes the following variables
in the main funtion:
    MODEL_SAVE_PATH 
    MODEL_LOAD_PATH

"""


import tensorflow as tf 
import cv2, glob
import numpy as np 
import random as rand

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from skimage import io
from sklearn.model_selection import train_test_split

# Class that can train a convolutional neural network using this projects dataset, load model,
# and evaluate a model.
class convNet(object):
    # Uses thresholding to remove the background from the hand training image and returns the 
    # Returns the thresholded image
    def preprocessImage(self, image):
        img = image

        # Threshold the image to remove the background and leave just the hand
        # threshold bounds determined through testing. Can probably still be improved
        ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

        # Use floodfill to fill in gaps in the hand that thresholding may have created
        # The thresholding already works quite well so this only helps a bit
        # May actually want to leave this out to simulate real world imperfections in the training data
        im_floodfill = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        img = img | im_floodfill_inv

        return img

    # Reads data out of the passed in directories and returns the the training images, training labels
    # testing images, and testing labels
    def readDataFromDir(self, train_dir):
        train_data = []
        train_label = []

        test_data = []
        test_label = []

        # Add all the training images and labels to their proper arrays
        for img in train_dir:
            current_img = io.imread(img, as_gray=True) # Read in image as grayscale

            # Preprocess the current image
            current_img = self.preprocessImage(current_img)

            # Append the preprocessed image to the training data set
            train_data.append(current_img)
            train_label.append(img[-5]) # The class label for each image is 6th to last character in the file name

        # Return as 4 numpy arrays
        return np.array(train_data), np.array(train_label)

    # Reads in the image from the passed in path, converts it to the format that can be directly
    # used by the model, and returns it
    def convertToModelFormat(self, img_path):
        # Read in image
        img = io.imread(img_path, as_gray=True)

        # Resize to proper size
        img = cv2.resize(img, (128, 128))

        # Convert to numpy array
        arr_img = np.array(img)

        # Add extra dimensions that the model wants
        arr_img = arr_img.reshape(1, arr_img.shape[0], arr_img.shape[1], 1)

        return arr_img

    # Train the model, save it to the passed in path, and return the trained model
    def trainModel(self, path):
        # File paths of training and testing images
        # Uses glob to grab all .png files in the directory
        train_img_dir = glob.glob("../training_images/*.png")

        # Get and display number and training and testing images
        num_train_imgs = len(train_img_dir)

        print(f"There are {num_train_imgs} training images")

        # Get the data and labels
        train_imgs, train_labels = self.readDataFromDir(train_img_dir)

        # Reshape the data into the proper shape
        train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2], 1)
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=6)

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

        # Compile the model with the optimizer, loss function, and desired metric
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

        # Callback checkpoint to save the model's weights whenever it improves over the best
        checkpoint = ModelCheckpoint(path, monitor='accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

        print("Right before the training")

        # Train the model with our data 
        model.fit(x = img_train, y = label_train, batch_size = 128, epochs = 10, validation_data=(img_test, label_test), callbacks=[checkpoint])

        self.evaluateModel(model, img_train, label_train)

        return model

    # Evaluate the passed in model with the images and labels
    def evaluateModel(self, model, images, labels):
        # Evaluate the model
        res = model.evaluate(images, labels, batch_size=128)

        print(f"Accuracy on evaluation set: {res[1]}, Loss on evaluation set {res[0]}")

    # Load the model from the passed in path and return it
    def loadModel(self, filename):
        print(f"Loading model from {filename}")

        model = load_model(filename)

        print("Model loaded!")

        return model


# Used for testing. Run this file directly to test the training, loading, and prediction
if __name__ == '__main__':
    # Create the object
    nn = convNet() 

    # Paths to save and load model
    MODEL_SAVE_PATH = "../models/new_model.h5"
    MODEL_LOAD_PATH = "../models/new_model.h5"

    # Test training the model
    model = nn.trainModel(MODEL_SAVE_PATH)

    # Make sure the model exists
    if(model == None):
        print("Model training failed")
        exit()

    # Test loading the model that was just trained and saved
    model = nn.loadModel(MODEL_LOAD_PATH)

    if(model == None):
        print("Model loading failed")
        exit()

    print("Hit any key while clicked on the image window to move to the next image")

    # Show how the model performs on 5 random samples from the dataset
    for i in range(0, 6):
        # Grab a random sample from the 2000 images of each of the 6 classes
        num = rand.randint(0, 2000)

        path = f'../training_images/{num}_{i}.png'

        # Convert the image to the proper format for the classifier to use
        img = nn.convertToModelFormat(path)

        # Get the prediction of the image from the classifier
        pred = model.predict(img)

        # Convert it to its actual class
        classes = pred.argmax(axis=-1)

        # Display the prediction
        print(f"Prediction: {classes[0]}")

        # Load and show the image that was classified
        img = cv2.imread(path)
        cv2.imshow("Image to classify", img)
        cv2.waitKey(0)