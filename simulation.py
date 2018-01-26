from perceptron import Perceptron
from plot import Plotter
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def simulation(x, test_range, step_size, file, n = 100, runs = 1000, dim = 2, learn_rate = 1):
    '''
    Function runs a series of simulations with the perceptron on a number or randomly generated feature vectors.
    Depending on which variable we are controlling for the simulations fix the values for dimensionality, number of points, and learning rate (c value)
    The variable that we control for will (x) will be initialized to the low end of the test range and incremented by the step size repeatedly.
    With each incrementation of the step size, we run the perceptron (with weights/bias always initialized to zero) 1000 times.
    After each single run, we record the results (i.e. number or perceptron iterations required for convergence) as a row in our dataframe
    The results are saved to a csv
    :param x: variable to control for, must be 'n', 'dim', or 'c'
    :param test_range: range of variable to test
    :param step_size: how to incrament the variable
    :param file: save destination for csv
    :return: N/A
    '''
    # check for invalid x
    if x not in ['n', 'c', 'dim']:
        raise ValueError('Invalid parameter x')

    (low, high) = test_range
    val = low
    data = []
    plot = Plotter()

    while val < high:
        # Increment independent variable
        if x == 'n':
            n = val
        elif x == 'c':
            learn_rate = val
        elif x == 'dim':
            dim = val
        # Run perceptron 1000 times each on a randomly generated set of feature vectors
        for i in range(runs):
            features = plot.generate_points(n, dim)
            labels = plot.generate_labels_linear(features)
            model = Perceptron(dim, zeros=False)
            iterations = model.train(features,labels, c=learn_rate)
            data.append([n, dim, learn_rate, iterations])
        val += step_size

    # Move data to pandas dataframe and save
    df = pd.DataFrame(data, columns=['n features', 'dimensions', 'c', 'iterations'])
    df.to_csv(file, sep=',', index=False)


def test_for_error_n(test_range, step_size, file, learn_rate = 1, dim = 2, runs = 100):
    (low, high) = test_range
    n = low
    data = []
    plot = Plotter()
    df = pd.DataFrame()
    while n < high:
        for i in range(runs):
            features = plot.generate_points(n, dim)
            labels = plot.generate_labels_linear(features)

            df['features'] = features.tolist()
            df['labels'] = labels.tolist()
            train, testing = train_test_split(df, test_size=.25)

            model = Perceptron(dim, zeros=True)
            model.train(to_array(train['features']), to_array(train['labels']), c=learn_rate)
            error = model.test_error(to_array(testing['features']), to_array(testing['labels']))
            data.append([n, error])

    df = pd.DataFrame(data, columns=['n', 'error'])
    df.to_csv(file, sep=',', index=False)

def to_array(arr):
    return np.array(arr.tolist())

if __name__ == "__main__":
    '''print('Simulation: Controlling for c....')
    simulation('c', test_range=(0.2,20), step_size=.2, file='data/raw_var_c_non-zero.csv')
    print('Simulation: Controlling for n....')
    simulation('n', test_range=(10, 1000), step_size=10, file='data/raw_var_n_non-zero.csv')
    print('Simulation: Controlling for dim....')
    simulation('dim', test_range=(2, 15), step_size=1, file='data/raw_var_dim_non-zero.csv')
    '''
    test_for_error_n(test_range=(10,100), step_size=10, file='data/raw_n_vs_error.csv')

    print('Complete')