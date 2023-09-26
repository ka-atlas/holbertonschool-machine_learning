#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    try:
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    
        nearest_centroid_indices = np.argmin(distances, axis=1)
    
        var = np.sum(distances[np.arange(len(X)), nearest_centroid_indices])

        return var
    except:
        return None
