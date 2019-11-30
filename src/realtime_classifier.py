"""
Author: Ryan O'Shea

Desciption:
Class defition and associated testing code for the realTimeDigitClassifier class.
Capable of calculating the average background, segmenting an image using that background 
to isolate the foreground, and passing that segmented into the classifier. Test code 
is in the main function should test various opencv functions to make they are working like
we want them to. 

********** IMPORTANT **********
If you want to change the model that gets loaded then change the file path in the 
nn.load_model call in the startClassifier() function.
********** IMPORTANT **********
"""


import cv2
import tensorflow
import keras 
from keras.models import load_model
import numpy as np

# Import class from model_training.py
from model_training import convNet

# Class for performing classification of how many fingers are currently being held up
# in an image. Uses the convNet to classify passed in images. Uses an average background
# and a number of opencv functions to segment the image of the hand to isolate it in the image.
class realTimeDigitClassifier(object):

	def __init__(self):
		self.bg = None # average background that will be subtracted from the image to isolate the hand in the image

	# Find the average background of the region of interest so it can be subtracted
	# Takes in an image of the background and the starting weight for the average
	# Doesn't return anything just updates the background
	def run_avg(self, image, aWeight):
		# initialize the background
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# compute weighted average, accumulate it and update the background
		cv2.accumulateWeighted(image, self.bg, aWeight)


	# Segment the passed in image and return the thresholded image and the contours of the hand
	def segment(self, image, threshold=25):
		# find the absolute difference between background and current frame
		diff = cv2.absdiff(self.bg.astype("uint8"), image)

		# threshold the diff image so that we get the foreground
		# Only grab the second item that the threshold function returns which is the 
		# actual thresholded image
		thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

		# get the contours in the thresholded image
		(cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# return None, if no contours detected
		if len(cnts) == 0:
			return
		else:
			# based on contour area, get the maximum contour which is the hand
			segmented = max(cnts, key=cv2.contourArea)
			return (thresholded, segmented)

	# Open a video feed, load the saved model, and classify images from the video feed with the model
	def startClassifier(self):
		# Load the saved model
		nn = convNet()

		model = nn.loadModel('../models/thresh_model_no_floodfill.h5')
		
		if(model == None):
			print("No model loaded!")
			exit()

		# Starting weight for the running average
		avgWeight = .5

		# Open the camera
		vid_cap = cv2.VideoCapture(0)

		#Check if the device was properly opened
		if not vid_cap.isOpened():
			raise Exception("Couldn't open camera")
			exit()

		# Used to save images to create training data
		i = 0

		# Rectangle params
		top, bottom, right, left = 80, 230, 450, 600
		rect_color = (0, 255, 0) # Green becasue it stands out
		rect_thickness = 3

		# Text params
		text_color = (0, 0, 255) # Red
		text_thickness = 1

		# Initilize number of frames to 0. The number will be used to capture the average background
		# during the beginning of the run 
		num_frames = 0

		# Start the video stream loop. Will quit if 'q' is pressed
		while(1):
			# Read picture in from the camera and store it in frame
			ret, frame = vid_cap.read()

			# Use flip to undo the mirroring of the image
			# Makes positioning yourself in the capture frame more intuitive
			frame = cv2.flip(frame, 1)

			# Make a clone of the frame to display later so we can alter frame
			clone = frame.copy()

			# Grab region of interest from the frame
			ROI = frame[top:bottom, right:left]

			# Convert the ROI to grayscale. Many of the opencv functions need the image to be in grayscale
			# because they don't work with more than 1 color dimension
			gray_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

			# Use Gaussian blur to smooth the image a bit
			gray_ROI = cv2.GaussianBlur(gray_ROI, (7,7), 0)

			# Use the first 30 frames to build the average background. There should be nothing but the background
			# in view during this to properly build an average background to subtract from the image
			if num_frames < 30:
				self.run_avg(gray_ROI, avgWeight)

				# While background is still being made tell the user to wait
				clone = cv2.putText(clone, 'WAIT', (right, top - 7), cv2.FONT_HERSHEY_COMPLEX, 1, text_color, text_thickness, cv2.LINE_AA)
			else: # Background building is done. Use it isolate the hand so it can be classified
				# Get the hand from the ROI 
				# hand contains both the thresholded images and the segmented portion
				# so we'll need to split it
				hand = self.segment(gray_ROI)

				# Draw instruction text on screen
				clone = cv2.putText(clone, 'Place hand in box', (right, top - 7), cv2.FONT_HERSHEY_COMPLEX, .5, text_color, text_thickness, cv2.LINE_AA)


				# Check whether the segmentation actually produced something
				if hand is not None:
					# Split the hand into the thresholded and segmented images
					# Thresholded will contain the actual thresholded image (the isolated hand)
					# Segmented will contain the contour around the hand which we will draw to the screen to
					# visualize what we're capturing as the hand
					(thresholded, segmented) = hand

					# Draw red contour lines around the hand in the image
					cv2.drawContours(image = clone, contours = [segmented + (right, top)], contourIdx = -1, color = (0, 0, 255), thickness = 2)

					# Display the thresholded image
					cv2.imshow("Thresholded ROI", thresholded)

					# Resize the image to the proper size for the cnn
					pred_img = cv2.resize(thresholded, (128, 128))

					# Turn it into a numpy array
					arr_img = np.array(pred_img)

					# Add two extra dimensions to the array to make 1x128x18x1
					# Not entirely sure why it needs the extra dimension but I get an error otherwise
					arr_img = arr_img.reshape(1, arr_img.shape[0], arr_img.shape[1], 1)

					# Predict what class the image is (How many fingers are being held up)
					pred = model.predict(arr_img)

					 # Convert it to its actual class
					classes = pred.argmax(axis=-1)

					# Draw the prediction on the screen
					clone = cv2.putText(clone, f'Prediction: {classes[0]}' , (right, bottom + 20), cv2.FONT_HERSHEY_COMPLEX, .75, text_color, text_thickness, cv2.LINE_AA)


			
			# Draw the rectangle on the frame around the ROI
			cv2.rectangle(clone, (left, top), (right, bottom), rect_color, rect_thickness)


			# Increment the number of frames we've captured
			num_frames += 1
			
			cv2.imshow("Video", clone)

			# Get what key was pressed
			keypress = cv2.waitKey(1) 

			# If the key pressed was q quit out of the program. 113 is the ascii value of q
			if keypress == 'q' or keypress == 113:
				break

			# Hit s to save an image
			if keypress == 's' or keypress == 115:
				resized = cv2.resize(thresholded, (128, 128))
				cv2.imwrite("../training_images/" + str(i) + "_2.png", resized)
				i += 1

				if(i > 2000):
					break;

		# Close all windows and free the camera
		vid_cap.release()
		cv2.destroyAllWindows()


# Main function for testing various parts of the code
if __name__ == '__main__':
	# Test video camera
	# Open the camera
	vid_cap = cv2.VideoCapture(0)

	#Check if the device was properly opened
	if not vid_cap.isOpened():
		raise Exception("Couldn't open camera")
		exit()

	ret, frame = vid_cap.read()

	# Display image to make we can get images
	cv2.imshow("Captured image", frame)

	print("Hit any button to move to the next image test.")

	# Wait until key is pressed
	cv2.waitKey(0)


	# Use flip to undo the mirroring of the image
	# Makes positioning yourself in the capture frame more intuitive
	frame = cv2.flip(frame, 1)

	# Display image 
	cv2.imshow("Flipped image", frame)

	# Wait until key is pressed
	cv2.waitKey(0)


	# Test drawing of the capture box on the image
	# Rectangle params
	top, bottom, right, left = 80, 230, 450, 600
	rect_color = (0, 255, 0) # Green becasue it stands out
	rect_thickness = 3

	# Draw the rectangle on the frame
	cv2.rectangle(frame, (left, top), (right, bottom), rect_color, rect_thickness)

	# Display image
	cv2.imshow("Rectangle on image", frame)

	# Wait until key is pressed
	cv2.waitKey(0)

	# Text params
	text_color = (0, 0, 255) # Red
	text_thickness = 1

	# Draw the prediction on the screen
	cv2.putText(frame, 'Test text!' , (right, bottom + 20), cv2.FONT_HERSHEY_COMPLEX, .75, text_color, text_thickness, cv2.LINE_AA)

	# Display image
	cv2.imshow("Text on image", frame)

	# Wait until key is pressed
	cv2.waitKey(0)