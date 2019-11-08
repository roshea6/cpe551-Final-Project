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

	# Subtract background from the image
	fgMask = backSub.apply(ROI)

	# frameRGB = frame[:,:,::-1]
	cv2.imshow('img', frame)
	cv2.imshow('ROI', fgMask)

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
