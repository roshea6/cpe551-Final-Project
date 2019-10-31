# cpe551-Final-Project
Final project for my Engineering Python course (CPE 551). A real time finger digit
classifier for determining how many fingers are held up on a hand. Also want to add in functionality to detect basic math symbols to turn it into a real time visual calculator.


Packages that will most likely be needed:
<ul>
	<li>OpenCV</li>
	<li>Numpy</li>
	<li>Keras</li>
	<li>Tensorflow</li>
	<li>Matplotlib</li>
	<li>os</li>
</ul

A dataset for finger digits was found on Kaggle (https://www.kaggle.com/koryakinp/fingers). Supplementary data may be generated using my computer's web camera and a python script using OpenCV. Data for the basic math symbols will have to be entirely generated using the same method if I am able to get to that part. The dataset will be used to train a constitutional neural network using Keras with a Tensorflow back end to create a classifier network. The classifier network will be used to to classify how many fingers are held up in a video stream through a laptop camera using OpenCV.

If I am able to get to the calculator portion of the project I will store the prediction results in variables and use those variables to perform basic math operations and return the result to the user. 
