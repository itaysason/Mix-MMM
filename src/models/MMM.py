import numpy as np
from src.models.logsumexp import logsumexp


class MMM:
    def __init__(self, k, init_params=None, epsilon=1e-4, max_iter=1e5):
        """
        :param k: number of topics
        :param m: number of mutations
        """
        self.k = k
        self.n, self.m = None, None
        self.e = None
        self.pi = None
        self.epsilon = epsilon
        self.max_iter = int(max_iter)
        self.set_params(init_params)
        self.data = None
        self.log_data = None

    def set_params(self, params):
        if params is None:
            return
        if 'e' in params.keys():
            self.e = params['e']
            _, self.m = self.e.shape
        if 'pi' in params.keys():
            self.pi = params['pi']
            self.n, self.m = self.pi.shape

    def expectation_step(self, value_list):
        log_likelihood = 0 if 'log_likelihood' in value_list else None
        log_expected_e = np.log(np.zeros((self.k, self.m))) if 'e' in value_list else None
        log_expected_pi = np.empty((self.n, self.k)) if 'pi' in value_list else None
        log_e = self.e
        for i in range(self.n):
            log_data = self.log_data[i]
            log_pi = self.pi[i]
            log_prob_topic_word = (log_e.T + log_pi).T
            log_prob_word = logsumexp(log_prob_topic_word, axis=0)
            curr_log_expected_e = log_prob_topic_word + log_data - log_prob_word

            if 'log_likelihood' in value_list:
                log_likelihood += np.inner(log_prob_word, self.data[i])

            if 'e' in value_list:
                np.logaddexp(curr_log_expected_e, log_expected_e, log_expected_e)

            if 'pi' in value_list:
                log_expected_pi[i] = logsumexp(curr_log_expected_e, axis=1)

        return log_expected_pi, log_expected_e, log_likelihood

    def maximization_step(self, log_expected_pi=None, log_expected_e=None):
        pi = log_expected_pi - logsumexp(log_expected_pi, axis=1, keepdims=True) if log_expected_pi is not None else self.pi
        e = log_expected_e - logsumexp(log_expected_e, axis=1, keepdims=True) if log_expected_e is not None else self.e
        return pi, e

    def fit(self, data):
        return self._fit(data, ['pi', 'e'])

    def refit(self, data):
        if self.e is None:
            raise Warning('e parameter is not given, using a random one instead')
        n, k = data.shape[0], self.k
        pi = np.empty((n, k))
        log_likelihood_change = 0
        for i in range(n):
            self.pi = None
            ll_improvement, b, c = self._one_sample_fit(data[i])
            log_likelihood_change += ll_improvement
            pi[i] = b[0]
            print(i)
        self.pi = pi
        self.n = data.shape[0]
        return log_likelihood_change, self.pi, self.e

    def _fit(self, data, params):
        self.data = data
        self.log_data = np.log(data)
        self.n, self.m = data.shape
        if self.e is None:
            self.e = np.random.dirichlet([0.5] * self.m, self.k)
        if self.pi is None:
            self.pi = np.random.dirichlet([0.5] * self.k, self.n)

        self.pi = np.log(self.pi)
        self.e = np.log(self.e)
        values_list = {'log_likelihood'}
        for param in params:
            values_list.add(param)
        log_expected_pi, log_expected_e, prev_log_likelihood = self.expectation_step(values_list)
        log_likelihood = prev_log_likelihood
        for iteration in range(self.max_iter):
            # maximization step
            self.pi, self.e = self.maximization_step(log_expected_pi, log_expected_e)

            # expectation step
            log_expected_pi, log_expected_e, log_likelihood = self.expectation_step(values_list)

            if log_likelihood - prev_log_likelihood < self.epsilon:
                break

            prev_log_likelihood = log_likelihood

        self.pi = np.exp(self.pi)
        self.e = np.exp(self.e)
        return log_likelihood

    def _one_sample_fit(self, x):
        if len(x.shape) == 1:
            x = np.array([x])
        return self._fit(x, ['pi'])

    def log_likelihood(self, data):
        self.data = data
        self.log_data = np.log(data)
        if self.e is None:
            self.e = np.random.dirichlet([0.5] * self.m, self.k)
        if self.pi is None:
            self.pi = np.random.dirichlet([0.5] * self.k, self.n)
        e = self.e.copy()
        pi = self.pi.copy()
        self.e = np.log(self.e)
        self.pi = np.log(self.pi)
        _, _, log_likelihood = self.expectation_step({'log_likelihood'})
        self.e = e
        self.pi = pi
        return log_likelihood

    def get_params(self):
        return {'pi': self.pi.copy(), 'e': self.e.copy()}
