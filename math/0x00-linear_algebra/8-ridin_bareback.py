#!/usr/bin/env python3

def mat_mul(mat1, mat2):
    if len(mat1[0])!=len(mat2):
        return None
    result = []

    return_mat = [[0 for x in range(len(mat2[0]))] for y in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                return_mat[i][j] += mat1[i][k] * mat2[k][j]

    return return_mat
