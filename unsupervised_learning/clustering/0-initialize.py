#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray):
        return None
    if type(k) is not int or X.shape[0] <= k or k <= 0:
        return None

    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    centroids = np.random.uniform(low=min,
                                  high=max,
                                  size=(k, X.shape[1]))

    return centroids
