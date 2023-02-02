from data_handler import bagging_sampler
import copy
import numpy as np
class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement this
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.estimators = []
        for i in range(self.n_estimator):
            self.estimators.append(copy.deepcopy(self.base_estimator))
            self.estimators_[i]._random_state = i



    def fit(self, X, y):
        # todo: implement this
        for estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.zeros((len(X), self.n_estimator))
        for i in range(self.n_estimator):
            predictions[:, i] = self.estimators[i].predict(X)
        predictions = np.mean(predictions, axis=1)
        return predictions