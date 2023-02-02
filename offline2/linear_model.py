
import numpy as np
class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.params = params
        # self.coef_, self.intercept_ = self._initialize_parameters()


    def fit(self, X, y):
        """
        Fit the linear regression model
        :param X: input data
        :param y: output data
        :return:
        """
        # todo: implement fitting algorithm
        self.X = X
        self.y = y
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

    def predict(self, X):
        """
        Predict the output for input data
        :param X: input data
        :return:
        """
        # todo: implement algorithm for prediction

        y_pred = np.dot(self.weight.T, X) + self.cons
        p = sigmoid(y_pred)
        p = p >= 0.5
        p = np.array(p, dtype=int)

        return p