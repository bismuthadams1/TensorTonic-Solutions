import math
import numpy as np

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    # create size X by X size grid

    # centre is (Size // 2, Size // 2) 
    CENTRE = size // 2
    
    def G(i, j) -> float:
        # get offsets
        x, y = np.abs(CENTRE - i), np.abs(CENTRE - j)
        return math.exp(-(x**2 + y**2)/(2*sigma**2))

    input_matrix = np.ndarray(size*size).reshape(size,-1)

    indices = np.indices((size,size))

    new_array = []

    total_weights = 0
    for i,j in zip(np.nditer(indices[0]), np.nditer(indices[1])):
        weight = G(i,j)
        new_array.append(weight)
        total_weights += weight

    unormalized = np.array(new_array).reshape(size,-1)

    normalized = unormalized / total_weights

    return normalized.tolist()
    

     