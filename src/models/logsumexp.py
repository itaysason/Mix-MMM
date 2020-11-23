import numpy as np


def logsumexp(a, axis=None, keepdims=False):
    """
    To reduce running time I implemented logsumexp myself, the scipy version has too much additional things I don't need
    :param a:
    :param axis:
    :param keepdims:
    :return:
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    output = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    output += a_max
    return output
