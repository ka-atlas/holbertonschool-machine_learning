#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    return [[mat1[j][i]+mat2[j][i] for j in range(len(mat1))]
                      for i in range(len(mat1[0]))]
