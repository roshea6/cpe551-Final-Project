import cv2
from matplotlib import pyplot as plt

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
	edges = cv2.Canny(ROI, 100, 200)

	# Subtract background from the image
	fgMask = backSub.apply(ROI)

	# Background subtraction then edge detection
	backsub_then_edges = cv2.Canny(fgMask, 100, 200)

	# Edge detection then background subtraction
	edges_then_backsub = backSub.apply(edges)
	


	# frameRGB = frame[:,:,::-1]
	cv2.imshow('img', frame)
	cv2.imshow('ROI', fgMask)
	cv2.imshow("Just edges", edges)
	cv2.imshow("Backsub then edges", backsub_then_edges)
	cv2.imshow("edges then backsub", edges_then_backsub)

	# Hit q to exit out of the video feed
	keyboard = cv2.waitKey(1)
	print(keyboard)
	if keyboard == 'q' or keyboard == 113:
		break

	# Hit s to save an image
	if keyboard == 's' or keyboard == 115:
		cv2.imwrite("../images/" + str(i) + ".png", fgMask)
		i += 1

vid_cap.release()
cv2.destroyAllWindows()
