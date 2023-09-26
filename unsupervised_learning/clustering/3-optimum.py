#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    """

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0 or kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    results = []
    d_vars = []

    k = 1
    C, clss = kmeans(X, k, iterations)
    var = variance(X, C)
    results.append((C, clss))
    d_vars.append(0.0)
  
    for k in range(kmin+1, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        d_vars.append(abs(var - variance(X, C)))

    return results, d_vars
