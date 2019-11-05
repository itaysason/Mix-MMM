import numpy as np
from scipy.special import logsumexp


class Mix:
    def __init__(self, num_clusters, num_topics, init_params=None, epsilon=1e-4, max_iter=1e5):
        """

        :param num_clusters: Positive integer, number of clusters to learn.
        :param num_topics: Positive integer, number of topics to learn.
        :param init_params: Dictionary with keys e, pi, w and values for initial parameters.
        :param epsilon: Positive small number, the stop criterion for fit.
        :param max_iter: Positive integer, maximum number of iterations for fit.
        """
        self.num_topics = num_topics
        self.num_clusters = num_clusters
        self.num_words = None
        self.e = None
        self.pi = None
        self.w = None
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
            if self.e.shape[0] != self.num_topics:
                raise ValueError('Initial parameter e is of shape {} but the first dimension has to be of size {}'.
                                 format(self.e.shape, self.num_topics))
            self.num_words = self.e.shape[1]
            self.e /= self.e.sum(1, keepdims=True)
        if 'pi' in params.keys():
            self.pi = params['pi']
            if self.pi.shape != (self.num_clusters, self.num_topics):
                raise ValueError('Initial parameter pi is of shape {} but the first dimension has to be of shape {}'.
                                 format(self.pi.shape, (self.num_clusters, self.num_topics)))
            self.pi /= self.pi.sum(1, keepdims=True)
        if 'w' in params.keys():
            self.w = params['w']
            if len(self.w) != self.num_clusters:
                raise ValueError('Initial parameter w is of shape {} but the first dimension has to be of shape {}'.
                                 format(self.pi.shape, (self.num_clusters, self.num_topics)))
            self.w /= self.w.sum()

    def set_data(self, data):
        self.data = data
        self.log_data = np.log(data)

    def pre_expectation_step(self):
        num_samples = self.data.shape[0]
        num_clusters, num_topics, num_words = self.num_clusters, self.num_topics, self.num_words
        log_likelihood = np.zeros((num_samples, num_clusters))
        log_expected_e = np.zeros((num_clusters, num_samples, num_topics, num_words))
        log_expected_pi = np.zeros((num_clusters, num_samples, num_topics))
        log_e = self.e
        for n in range(self.data.shape[0]):
            curr_log_b = self.log_data[n]
            curr_b = self.data[n]
            for l, log_pi in enumerate(self.pi):
                log_prob_topic_word = (log_e.T + log_pi).T
                log_prob_word = logsumexp(log_prob_topic_word, axis=0)
                log_likelihood[n, l] = np.inner(log_prob_word, curr_b)

                log_expected_e[l, n] = log_prob_topic_word + curr_log_b - log_prob_word

                log_expected_pi[l, n] = logsumexp(log_expected_e[l, n], axis=1)

        return log_expected_pi, log_expected_e, log_likelihood

    def expectation_step(self):
        expected_pi_sample_cluster, expected_e_sample_cluster, likelihood_sample_cluster = self.pre_expectation_step()

        likelihood_sample_cluster += self.w
        tmp = logsumexp(likelihood_sample_cluster, 1, keepdims=True)
        log_likelihood = np.sum(tmp)
        likelihood_sample_cluster -= tmp
        expected_pi_sample_cluster += likelihood_sample_cluster.T[:, :, np.newaxis]
        expected_e_sample_cluster += likelihood_sample_cluster.T[:, :, np.newaxis, np.newaxis]
        log_expected_pi = logsumexp(expected_pi_sample_cluster, 1)
        log_expected_e = logsumexp(expected_e_sample_cluster, (0, 1))
        expected_w = logsumexp(likelihood_sample_cluster, 0)
        return expected_w, log_expected_pi, log_expected_e, log_likelihood

    def maximization_step(self, log_expected_w, log_expected_pi, log_expected_e, params):
        if 'w' in params:
            w = log_expected_w - logsumexp(log_expected_w)
        else:
            w = self.w
        if 'pi' in params:
            pi = log_expected_pi - logsumexp(log_expected_pi, axis=1, keepdims=True)
        else:
            pi = self.pi
        if 'e' in params:
            e = log_expected_e - logsumexp(log_expected_e, axis=1, keepdims=True)
        else:
            e = self.e
        return w, pi, e

    def refit(self, data):
        """
        Fitting only the clusters (pi) and the weights (w).
        :param data:
        :return:
        """
        if self.e is None:
            raise ValueError('e was not set, can not refit')
        return self._fit(data, ['w', 'pi'])

    def fit(self, data):
        """
        Fitting all the parameters of the model.
        :param data:
        :return:
        """
        return self._fit(data, ['w', 'pi', 'e'])

    def _fit(self, data, params):
        self.set_data(data)
        if self.num_words is None:
            self.num_words = data.shape[1]
        else:
            if self.num_words != data.shape[1]:
                raise ValueError('data and the given topics shapes, {}, {}, do not match', data.shape, self.e.shape)
        if self.e is None:
            self.e = np.random.dirichlet([0.5] * self.num_words, self.num_topics)
        if self.pi is None:
            self.pi = np.random.dirichlet([0.5] * self.num_topics, self.num_clusters)
        if self.w is None:
            self.w = np.random.dirichlet([2] * self.num_clusters)

        self.pi = np.log(self.pi)
        self.w = np.log(self.w)
        self.e = np.log(self.e)

        expected_w, log_expected_pi, log_expected_e, prev_log_likelihood = self.expectation_step()
        log_likelihood = prev_log_likelihood
        for iteration in range(self.max_iter):
            print(log_likelihood)
            # maximization step
            self.w, self.pi, self.e = self.maximization_step(expected_w, log_expected_pi, log_expected_e, params)

            # expectation step
            expected_w, log_expected_pi, log_expected_e, log_likelihood = self.expectation_step()

            if log_likelihood - prev_log_likelihood < self.epsilon:
                break

            prev_log_likelihood = log_likelihood

        self.pi = np.exp(self.pi)
        self.e = np.exp(self.e)
        self.w = np.exp(self.w)
        return log_likelihood

    def log_likelihood(self, data):
        max_iter, self.max_iter = self.max_iter, 0
        ll = self.fit(data)
        self.max_iter = max_iter
        return ll

    def predict(self, data):
        """

        :param data: ndarray (N, num_words) of non-negative integers.
        :return clusters: ndarray (N,) of integers representing the clusters.
        :return topics: ndarray (N, num_topics) of non-negative integers, counts of topics in the data.
        :return probabilities: ndarray (N,), probabilities[i] = log(Pr[sample[i], topics[i], clusters[i]])
        """
        num_samples = len(data)
        clusters = np.zeros(num_samples, dtype='int')
        probabilites = np.log(np.zeros(num_samples))
        topics = np.zeros((num_samples, self.num_topics), dtype='int')
        log_e = np.log(self.e)
        log_w = np.log(self.w)
        for cluster in range(self.num_clusters):
            curr_pi = np.log(self.pi[cluster])
            pr_topic_word = (log_e.T + curr_pi).T
            likeliest_topic_per_word = np.argmax(pr_topic_word, axis=0)
            for sample in range(num_samples):
                curr_prob = log_w[cluster]
                curr_topic_counts = np.zeros(self.num_topics, dtype='int')
                curr_word_counts = data[sample]
                for word in range(len(curr_word_counts)):
                    curr_topic_counts[likeliest_topic_per_word[word]] += data[sample, word]
                    curr_prob += data[sample, word] * pr_topic_word[likeliest_topic_per_word[word], word]
                if curr_prob > probabilites[sample]:
                    probabilites[sample] = curr_prob
                    clusters[sample] = cluster
                    topics[sample] = curr_topic_counts
        return clusters, topics, probabilites
    
    def expected_topics(self, data):
        self.pi = np.log(self.pi)
        self.w = np.log(self.w)
        self.e = np.log(self.e)
        self.set_data(data)
        expected_pi_sample_cluster, expected_e_sample_cluster, likelihood_sample_cluster = self.pre_expectation_step()
        likelihood_sample_cluster += self.w
        likelihood_sample_cluster -= logsumexp(likelihood_sample_cluster, 1, keepdims=True)
        expected_pi_sample_cluster += likelihood_sample_cluster.T[:, :, np.newaxis]
        expected_e_sample_cluster += likelihood_sample_cluster.T[:, :, np.newaxis, np.newaxis]
        log_expected_topics = logsumexp(expected_pi_sample_cluster, 0)
        self.pi = np.exp(self.pi)
        self.e = np.exp(self.e)
        self.w = np.exp(self.w)
        return np.exp(log_expected_topics)
    
    def get_params(self):
        return {'pi': self.pi.copy(), 'w': self.w.copy(), 'e': self.e.copy()}
