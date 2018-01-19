# homebrew_perceptron

**Perceptron that classifies n-dimensional linearly seprable data. Matplotlib for 3D graph example. **

This code is an implementation of Dr.Frank Rosenblatt's perceptron algorithm. Consider the following image....

![](https://raw.githubusercontent.com/gonzodeveloper/homebrew_perceptron/master/img/LTU.png)

This depicts an algorithm whose output is determined by a series of inputs(x), weights(w) and bias(w0). We can use this for classification by repeatedly inputing labeled feature vectors, and adjusting the weights and bias accordingly based on whether the output y matches the given labels. This algorithm is given as...

> while error
> 	error = False
>	for idx, x in features
>		if sgn(x, weights) + bias != labels[i]
>			error = True
>			weights += x * labels[i]
> 			bias += labels[i]

This is implemented in perceptron.py as part of the train function, which also limits the iteration to a given number of epochs.

To test this I have created a program to generate random points in 3D space and labeling them either according to some arbitray linear seperability or in an XOR pattern about the Y-Axis. The program then trains a perceptron on each set of points and labels. At each step of training we print the figure. The points are colored according to their predetermined labels, and a plane of separation as determined by the weights and bias is shown for each step. 

To illustrate this process I have collated the images into animated gifs in order to show the percptron converging (or failing to do so in the XOR case) on a proper boundry of separation.

**LINEARLY SEPERABLE DATA**

![](https://raw.githubusercontent.com/gonzodeveloper/homebrew_perceptron/master/img/lin.gif)

**XOR DATA**

![](https://raw.githubusercontent.com/gonzodeveloper/homebrew_perceptron/master/img/xor.gif)
