Shape Classifier (Deep Learning) : Is that image a triangle or a square or a circle using CNN/Deep Learning.
==================

There are a lot of different types of shapes and it is important to be able to differentiate between them.

In this project, We have created deep convolution neural network to classify different drawing images and tag them as a triangle, square or a circle.

First we are training our model using training dataset with 3 main classes.

After that we will save our trained model into a h5 file.

We will use this pretrained model to predict our custom images(single) or image set(multiple images) from test dataset.

We are visulizing accuracy and loss of validation after each epoch using matplotlib library.

There is another dataset to practice with four categorical shapes. 


Installation
==================

To start with project just follow the few steps 

	$ git clone https://github.com/keyurr2/shape-classifier-cnn.git
	$ pip install -r requirements.txt
	
This will install python libraries required to start with Deep Learning like Tensorflow and Keras

NOTE: We are using Python 3 in this project.


How to run this project
==================================================
The first step is to train the model using training dataset. (Though it's already trained)
	
	$ python cnn.py

It will take some time and we can find the logs for that.

After that we can run our prediction on single image file or on test/validation image dataset

To give single image of shape stored on your computer and predict use script as below

	$ python predict.py --image <image-path>

To run classification on test dataset use script as below

	$ python predict.py --testdata

To run classification on validation dataset use script as below

	$ python predict.py --validationdata

You can find the confusion matrix and accuracy of model after prediction.

Authors
==================

* **Keyur Rathod (keyur.rathod1993@gmail.com)**

License
==================

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
