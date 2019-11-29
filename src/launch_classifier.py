"""
Author: Ryan O'Shea

Description:
Main driver code for the realTimeDigitClassifier. Only imports 
the class, creates an object, and then calls the class function to start the classifier.
If any of the functionalities need to be tested then the realtime_classifier.py and
model_training.py files should be run individually.
"""

# Import classifier from realtime_classifier file
from realtime_classifier import realTimeDigitClassifier

# Create a realTimeDigitClassifier object
classifier = realTimeDigitClassifier()

# Start the classifier
classifier.startClassifier()