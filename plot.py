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


def generate_points(n):
    '''
    Creates an array of random but linearly separable points (i.e. feature vectors) in 3 dimensional space.
    X,Y,Z float values are restricted to the range (-1, 1)
    Points are given labels, either 1 or -1.
    :param n: number of points to create
    :return: features: an array of points
            labels: an array of labels, either 1 or -1
    '''

    return np.random.uniform(low=-1, high=1, size=(n, 3))


def generate_labels_linear(features):
    '''

    :param features:
    :return:
    '''
    # Normal vector for linear separation of points
    norm = np.random.uniform(low=-1, high =1, size=3)

    labels = np.empty(shape=len(features))

    for i, x in enumerate(features):
        # Determine which side of our normal the point is on
        lab = np.dot(norm, x)
        if lab > 0:
            labels[i] = 1
        else:
            labels[i] = -1
    return labels

def generate_labels_xor(features):
    '''

    :param features:
    :return:
    '''
    labels = np.empty(shape=len(features))
    for i, x in enumerate(features):
        if x[0] >= 0 and x[2] >= 0:
            labels[i] = 1
        elif x[0] >= 0 and x[2] < 0:
            labels[i] = -1
        elif x[0] < 0 and x[2] >= 0:
            labels[i] = -1
        elif x[0] < 0 and x[2] < 0:
            labels[i] = 1
    return labels

def plot_graph(model, features, labels, epochs, dir):
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
    for weights, bias in model.train(features, lab_lin, epochs=epochs):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Iteration: {}\nWeights: {}\n Bias: {}'.format(count, weights, bias))
        plt.ioff()
        ax.set_zlim3d(-1,1)
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)

        # Plot points
        for i, x in enumerate(features):
            ax.scatter(x[0], x[1], x[2], c=colors[i], marker=shapes[i])

        # The equation of the plane is given by ax + bx + cx + d = 0
        # We know that weights = < a, b, c > and bias = d
        # Use this to solve for z and plot
        [xx, yy] = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
        z = (-weights[0] * xx - weights[1] * yy - bias) / weights[2]
        ax.plot_surface(xx, yy, z, color='g', alpha=0.5)
        ax.view_init(10, 90)
        plt.savefig('{}/itration{}'.format(dir, str(count).zfill(2)))
        count += 1


if __name__ == "__main__":

    features = generate_points(50)
    lab_lin = generate_labels_linear(features)
    lab_xor = generate_labels_xor(features)

    lin_model = Perceptron(inputs=3)
    xor_model = Perceptron(inputs=3)

    plot_graph(lin_model, features, lab_lin, epochs=50, dir="lin_img")
    plot_graph(xor_model, features, lab_xor, epochs=50, dir="xor_img")


