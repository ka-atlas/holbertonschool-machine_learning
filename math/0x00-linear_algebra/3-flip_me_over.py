def matrix_transpose(matrix):
    if type(matrix[0])!= list:
        return "given matrix is not 2D"
    else:
        new_matrix = [[matrix[j][i] for j in range(len(matrix))]
                      for i in range(len(matrix[0]))]
    return new_matrix
