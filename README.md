# cpe551-Final-Project (Real Time Finger Digit Classifier)
Final project for my Engineering Python course (CPE 551). A real time finger digit classifier for determining how many fingers are held up on a hand. Uses a convolutional neural network trained on a dataset I created to determine how many fingers are being held up on a hand in an image.


## Required Python Packages:
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
</ul>

## Background
There were existing datasets out there for hands holding up different numbers of digits but I couldn't seem to get good results using existing datasets so I decided to generate my own. I created 2000 images for each number of fingers being held up to train my neural network on. I uploaded my dataset to kaggle here (https://www.kaggle.com/roshea6/finger-digits-05) so others could use it. My dataset already has the images segmented to fully isolate the hand in the image to make it more beginner friendly. The dataset was used to train a convolutional neural network using Keras with a Tensorflow back end to create a 6 class classifier network. The classifier network is used to classify how many fingers are held up in a video stream through a laptop camera using OpenCV.

## Repository Structure
<ul>
	<li>src
		<ul>
			<li>Contains the python files for the various classes and driver code for the real time classifier. Launch classifier is the main file to actually start the code.</li>
		</ul>
	</li>
	<li>models
		<ul>
			<li>Contains the models I trained ad a .h5 model. These models can be loaded using the keras load_model function. They are zipped up so I can actually have them on github. Extract them to the models folder if you want to use them.</li>
		</ul>
	</li>
	<li>training_images
		<ul>
			<li>Contains the images used to train the models in the models folder. These can be used to train your own model. There are 12006 128x128 images of a hand showing the numbers 0 through 5 as fingers held up.</li>
			![alt text][media/dataset_banner.png]
		</ul>
	</li>
	<li>media
		<ul>
			<li>Contains the demo images and videos for the repository and kaggle dataset.</li>
		</ul>
	</li>
</ul>

## How to use
<ol>
	<li>Clone this repository</li>
	<li>Install the required packages listed above</li>
	<li>Extract the .h5 files in the models folder to the models folder</li>
	<li>Test any of the functionality you want to by running the file directly</li>
	<li>Make sure your camera is facing a wall or something that will be easy to build an average background with</li>
	<li>Run launch_classifier.py</li>
	<li>Wait until the words above the box say "Place hand in box"</li>
	<li>Place you hand in the box on the video feed to see what the classifier outputs!</li>
</ol>

## How to test functionality
<ol>
	<li>Run either realtime_classifier.py or model_training.py. Their main functions contain test code to test all the different functions of the classes.</li>
	<li>Ensure that all the functionality is working as intended. There should be printed instructions and images that pop up to make sure everything is working</li>
	<li>If everything looks good then you should good to go!</li>
</ol>