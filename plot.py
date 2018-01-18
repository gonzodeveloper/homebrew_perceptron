# Kyle Hart
# 14 January 2018
#
# Project: homebrew_perceptron
# File: plot.py
# Description: Main program, generates a list of labeled 3 dimensional points and uses the Perceptron to classify

import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generatePoints(n):
    '''
    Creates an array of random but linearly separable points (i.e. feature vectors) in 3 dimensional space.
    X,Y,Z float values are restricted to the range (-1, 1)
    Points are given labels, either 1 or -1.
    :param n: number of points to create
    :return: features: an array of points
            labels: an array of labels, either 1 or -1
    '''
    # Normal vector for linear separation of points
    norm = np.random.uniform(low=-1, high =1, size=3)

    features = np.empty(shape=(n, 3))
    labels = np.empty(shape=(n, 1))

    for i in range(n):
        # Create random point
        point = np.random.uniform(low=-1, high=1, size=3)
        # Determine which side of our normal the point is on
        lab = np.dot(norm, point)

        if lab > 0:
            features[i] = point
            labels[i] = 1
        else:
            features[i] = point
            labels[i] = -1

    return features, labels

if __name__ == "__main__":

    features, labels = generatePoints(20)
    model = Perceptron(inputs=3)

    # Get map our labels to shapes and colors for the graph
    shapes = []
    colors = []
    for x in labels:
        if x == 1:
            shapes.append("^")
            colors.append("r")
        else:
            shapes.append("o")
            colors.append("b")

    count = 1
    # For each iteration the perceptron runs, we plot the points in 3d space as well as the plane of separation
    # which is given by the weights and bias.
    for weights, bias in model.train(features, labels, epochs=20):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Iteration: {}\nWeights: {}\n Bias: {}'.format(count, weights, bias))

        # Plot points
        for i, x in enumerate(features):
            ax.scatter(x[0], x[1], x[2], c=colors[i], marker=shapes[i])

        # The equation of the plane is given by ax + bx + cx + d = 0
        # We know that weights = < a, b, c > and bias = d
        # Use this to solve for z and plot
        [xx, yy] = np.meshgrid(np.arange(-1,1,0.1), np.arange(-1,1,0.1))
        z = (-weights[0]*xx - weights[1]*yy - bias)/weights[2]
        ax.plot_surface(xx, yy, z, color ='g')

        count += 1
        plt.savefig('img/itration{}'.format(count))

