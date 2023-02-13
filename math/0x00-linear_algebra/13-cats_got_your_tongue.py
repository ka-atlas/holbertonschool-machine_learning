
#!/usr/bin/env python3

import numpy as np

def np_cat(mat1, mat2, axis=0):
    """
    Using numpy, concatenate two mnatrices along a specified axis
    """
    return_array = np.concatenate((mat1, mat2), axis)
    return 
