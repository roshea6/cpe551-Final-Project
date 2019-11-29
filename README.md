# cpe551-Final-Project (Real Time Finger Digit Classifier)
Final project for my Engineering Python course (CPE 551). A real time finger digit classifier for determining how many fingers are held up on a hand. 


Packages that will most likely be needed:
<ul>
	<li>cv2</li>
	<li>numpy</li>
	<li>keras</li>
	<li>tensorflow</li>
	<li>matplotlib</li>
	<li>glob</li>
	<li>skimage</li>
	<li>sklearn</li>
	<li>random</li>
</ul

I couldn't seem to get good results using existing datasets so I decided to generate my own. I created 2000 images for each of number of fingers being held up to train my neural network on. I uploaded my dataset to kaggle here (https://www.kaggle.com/roshea6/finger-digits-05) so others could use it. My dataset already has the images segmented to fully isolate the hand in the image to make it more beginner friendly. The dataset will be used to train a convolutional neural network using Keras with a Tensorflow back end to create a classifier network. The classifier network will be used to to classify how many fingers are held up in a video stream through a laptop camera using OpenCV.

