#!/usr/bin/env python3
"""Sigma Equation"""


def summation_i_squared(n):
    """Addition With Sigma"""
    if n>=1:
        return int((n*(n+1)*(2*n+1))/6)
    else:
        return None
