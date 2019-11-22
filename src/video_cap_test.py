import cv2
from matplotlib import pyplot as plt
import tensorflow
import keras 
from keras.models import load_model
import numpy as np

model = load_model('../models/thresh_model.h5')

# Open the camera
vid_cap = cv2.VideoCapture(0)

#Check if the device was properly opened
if not vid_cap.isOpened():
	raise Exception("Couldn't open camera")

# Background subtractor using mixture of gaussian method
backSub = cv2.createBackgroundSubtractorMOG2()

i = 0

# Rectangle params
start_point = (50, 200) # Upper left corner
end_point = (200, 350) # Bottom right corner
color = (0, 255, 0)
thickness = 3

while(1):
	# Read picture
	ret, frame = vid_cap.read()

	# Grab region of interest from the frames
	ROI = frame[200:350, 50:200]

	# Plot rectangle on the frame to show the ROI
	frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

	# Just edge detection
	edges = cv2.Canny(ROI, 150, 200)

	# Subtract background from the image
	# fgMask = backSub.apply(ROI)

	# # Background subtraction then edge detection
	# backsub_then_edges = cv2.Canny(fgMask, 100, 200)

	# # Edge detection then background subtraction
	# edges_then_backsub = backSub.apply(edges)

	gray_ROI = cv2.cvtColor(ROI, cv2.COLOR_RGB2GRAY)

	# cv2.imshow("Thresh", gray_ROI)
	# cv2.waitKey(0)

	ret, img = cv2.threshold(gray_ROI, 20, 255, cv2.THRESH_BINARY_INV)

	# cv2.imshow("Thresh", img)
	# cv2.waitKey(0)

	# im_floodfill = img.copy()
	# h, w = img.shape[:2]
	# mask = np.zeros((h+2, w+2), np.uint8)
	# cv2.floodFill(im_floodfill, mask, (0,0), 255)
	# im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	# img = img | im_floodfill_inv


	# frameRGB = frame[:,:,::-1]
	cv2.imshow('img', frame)
	cv2.imshow('ROI', gray_ROI)
	# cv2.imshow("Just edges", edges)
	cv2.imshow("Thresh", img)
	# cv2.imshow("Backsub", fgMask)
	# cv2.imshow("edges then backsub", edges_then_backsub)

	pred_img = cv2.resize(img, (128, 128))

	arr_img = np.array(pred_img)

	arr_img = arr_img.reshape(1, arr_img.shape[0], arr_img.shape[1], 1)

	out = model.predict(arr_img)

	print(out)

	# Hit q to exit out of the video feed
	keyboard = cv2.waitKey(1)

	if keyboard == 'q' or keyboard == 113:
		break

	# Hit s to save an image
	if keyboard == 's' or keyboard == 115:
		cv2.imwrite("../test_images/" + str(i) + "_mine.png", edges)
		i += 1

vid_cap.release()
cv2.destroyAllWindows()
