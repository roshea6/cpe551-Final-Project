import cv2, keras, tensorflow
from skimage import io
import numpy as np 
from keras.models import load_model

model = load_model('../models/thresh_model.h5')

img = io.imread('../test_images/2.png', as_gray=True)

img = cv2.resize(img, (128, 128))

cv2.imshow('Thresh', img)
cv2.waitKey(0)

ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('Thresh', img)
cv2.waitKey(0)

im_floodfill = img.copy()
h, w = img.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
img = img | im_floodfill_inv

cv2.imshow('Thresh', img)
cv2.waitKey(0)

arr_img = np.array(img)

arr_img = arr_img.reshape(1, 128, 128, 1)

print(model.predict(arr_img))