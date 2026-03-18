import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)

    new_A = []
    for element in np.nditer(A, order= 'F'):
        new_A.append(element)
        
    new_A = np.asarray(new_A)
    new_A= new_A.reshape((A.shape[1], A.shape[0]))

    return new_A