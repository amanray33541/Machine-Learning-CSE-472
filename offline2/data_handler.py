import numpy as np
import  pandas as pd
import random
def load_dataset():
    """
       function for reading data from csv
       and processing to return a 2D feature matrix and a vector of class
       :return:
       """
    # todo: implement
    # Load the data
    data = pd.read_csv('data_banknote_authentication.csv', header = None)
    data = data.values

    X = data[:, :-1]
    y = data[:, -1]
    X  = X[1:,:]
    y = y[1:]
    X = X.astype(float)
    y = y.astype(float)
    return X, y





def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    dataSize = y.shape[1]
    test_size = int(test_size * dataSize)
    arr = np.arange(0, dataSize, dtype=int)
    if shuffle:
        np.random.shuffle(arr)
    X_test = X[:, arr[0:test_size]]
    y_test = y[:, arr[0:test_size]]
    X_train = X[:, arr[test_size:dataSize]]
    y_train = y[:, arr[test_size:dataSize]]

    # X_train, y_train, X_test, y_test = None,None,None,None
    return X_train, y_train, X_test, y_test

def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # # todo: implement
    # X_sample, y_sample = None, None
    # assert X_sample.shape == X.shape
    # assert y_sample.shape == y.shape
    # return X_sample, y_sample

    X_sample, y_sample = None, None
    n_samples = X.shape[0]
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sample = X[idx]
    y_sample = y[idx]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample

X,y = load_dataset()
print(X,y)