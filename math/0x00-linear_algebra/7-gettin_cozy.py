#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    return_matrix = []
    if(axis == 0):
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat1:
            return_matrix.append(row.copy())
        for row in mat2:
            return_matrix.append(row.copy())
        return return_matrix
    if(axis == 1):
        if len(mat1) != len(mat2):
            return None
        for col in range(len(mat1)):
            return_matrix.append(mat1[col].copy() + mat2[col].copy())
        return(return_matrix)
