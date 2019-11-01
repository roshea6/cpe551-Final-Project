import cv2
from matplotlib import pyplot as plt

# Open the camera
vid_cap = cv2.VideoCapture(0)

#Check if the device was properly opened
if not vid_cap.isOpened():
	raise Exception("Couldn't open camera")

i = 0

while(1):
	# Read picture
	ret, frame = vid_cap.read()

	# frameRGB = frame[:,:,::-1]
	cv2.imshow('img', frame)

	# Hit q to save an image
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.imwrite("../images/" + str(i) + ".png", frame)
		i += 1

vid_cap.release()
cv2.destroyAllWindows()
