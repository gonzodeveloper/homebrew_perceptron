# Kyle Hart
# 14 January 2018
#
# Project: homebrew_perceptron
# File: perceptron.py
# Description: a perceptron that classifies linearly separable points

import numpy as np

class Perceptron:

    def __init__(self, inputs, zeros=False):
        '''
        Contructor gives perceptron zeros for weights and bias
        :param inputs: number of dimensions in feature vectors
        :param zeros: if true then weights and bias are set to zero, otherwise they are assigned random values
        '''
        if zeros:
            self.weights = np.zeros(inputs)
            self.bias = np.zeros(1)
        else:
            self.weights = np.random.uniform(low=-1, high=1, size=inputs)
            self.bias = np.random.uniform(low=-1, high=1, size=1)


    def activation(self, x):
        '''
        Activation function (weights DOT x) + bias. Returns 1 or -1 based on positive or negative ans.
        :param x: feature vector
        :return: -1 or 1 based on return of activation
        '''
        return 1 if np.dot(self.weights, x) + self.bias >= 0 else -1



    # Temporarily NOT a generator function, will NOT work in plot.py unless uncomment line 59
    def train(self, features, labels, epochs=10000, c=1):
        '''
        Generator Function.
        Yields a series of tuples, each containing alist of weights and bias for each increment of the perceptron.
        Based on Rosenblatt's single layer perceptron algorithm
        :param features: numpy array of feature vectors
        :param labels: numpy array of labels
        :param epochs: number of training iterations for the perceptron
        :param c: constant multiplier for increments of weight vecotr
        :yields array of weights and a bias constant for each iteration of perceptron
        :return: number of iterations to convergence
        '''
        error_detected = True
        itr = 0
        while error_detected and itr < epochs:
            itr += 1
            error_detected = False
            for i, x in enumerate(features):
                if self.activation(x) != labels[i]:
                    error_detected = True
                    self.weights += c*(x * labels[i])
                    self.bias += c*labels[i]
            # yield self.weights, self.bias
        return itr

    def test_error(self, features, labels, epochs=10000, c=1):
        error_count = 0
        for i, x in enumerate(features):
            if self.activation(x) != labels[i]:
                error_count += 1
        return error_count/len(features)