import time
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, k, kFinal):
        self.k = k
        self.log_likelihood = -np.inf
        self.kFinal = kFinal

    def fit(self, X, n_iter=30):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.means = X[np.random.choice(n_samples, self.k, replace=False), :]
        # print(self.means.shape)
        self.covariances = np.array([np.eye(n_features) for _ in range(self.k)])
        self.weights = np.ones(self.k) / self.k

        if n_features == 2 and self.kFinal:
            plt.ion()
            fig = plt.figure()

        for i in range(n_iter):
            # E-step
            responsibilities = self._compute_responsibilities(X)

            # M-step
            self.weights = np.mean(responsibilities, axis=0)
            self.means = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0, keepdims=True).T
            for j in range(self.k):
                x_mu = X - self.means[j]
                self.covariances[j] = np.dot(
                    x_mu.T, x_mu * responsibilities[:, j, np.newaxis]) / np.sum(responsibilities[:, j])

            new_log_likelihood = self._compute_log_likelihood(X)
            # convergence
            if np.abs(new_log_likelihood - self.log_likelihood) < 1e-3:
                break
            self.log_likelihood = new_log_likelihood

            if n_features == 2 and self.kFinal:
                plt.clf()
                plt.scatter(X[:, 0], X[:, 1], color='#68BBE3')
                x, y = np.mgrid[X[:, 0].min():X[:, 0].max():.01,
                       X[:, 1].min():X[:, 1].max():.01]
                pos = np.empty(x.shape + (2,))
                pos[:, :, 0] = x
                pos[:, :, 1] = y
                for i in range(self.k):
                    rv = multivariate_normal(mean=self.means[i], cov=self.covariances[i])
                    plt.contour(x, y, rv.pdf(pos))

                # clear previous drawing
                # plt.savefig('gmm.png'+i)
                fig.canvas.draw()
                # fig.suptitle("GMM with k = "+k )
                fig.canvas.flush_events()
                time.sleep(0.001)

        return self.means, self.covariances, self.weights, self.log_likelihood

    def _compute_responsibilities(self, X):
        responsibilities = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            responsibilities[:, k] = self.weights[k] * \
                                     multivariate_normal.pdf(
                                         X, self.means[k], self.covariances[k])
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities

    def _compute_log_likelihood(self, X):
        n_samples, n_features = X.shape
        k = self.k

        loglikelihood = 0
        for i in range(n_samples):
            sample_loglikelihood = 0
            for j in range(k):
                pdf = multivariate_normal.pdf(X[i], mean=self.means[j], cov=self.covariances[j], allow_singular=True)
                sample_loglikelihood += self.weights[j] * pdf
            loglikelihood += np.log(sample_loglikelihood)

        return loglikelihood


if __name__ == "__main__":
    data = np.loadtxt('data2D.txt')

    loglikelihoods = []
    k = 3
    for i in range(1, k + 1):
        gmm = GMM(k=i, kFinal=False)
        means, covariances, weights, likelihood = gmm.fit(data)
        loglikelihoods.append(likelihood)
        print("k", " = ", i, " ", likelihood)

    plt.plot(range(1, k + 1), loglikelihoods)
    plt.xlabel('k')
    plt.ylabel('log likelihood')
    plt.show()

    # k*
    k_star = np.argmax(loglikelihoods) + 1
    print("k* = ", k_star)
    gmm2 = GMM(k=k_star, kFinal=True)
    means, covariances, weights, likelihood = gmm2.fit(data)
    # print("means = ", means)
    # print("covariances = ", covariances)
    # print("weights = ", weights)


bf['count'] = 1
bf_hour_list = []
for hour in bf.hour.sort_values().unique():
    df_hour_list.append(bf.loc[df.hour == hour, ['start_lat', 'start_lon', 'count']].groupby(['start_lat', 'start_lon']).sum().reset_index().values.tolist())




