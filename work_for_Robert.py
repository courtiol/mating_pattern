import numpy as np
from itertools import product
import math
from likelihood_n_by_m import compute_h, computeA

def computeGeneralRecurrence1(Q, P):
    """
    This method computes the recurrence A[Q] = 1/* ( A[Q-E11] + ... A[Q-E_mn])
    :param Q: multidimensional numpy.array of type <class 'numpy.ndarray'> with coefficients of <class 'numpy.int64'>
    :param P: multidimensional numpy.array of type <class 'numpy.ndarray'> with coefficients of <class 'numpy.float64'>
    :return: the recurrence term A[Q] of type <class 'numpy.float64'>
    """
    nrows = np.shape(Q)[0]
    ncols = np.shape(Q)[1]
    sizeShape = ncols * nrows

    shape = (np.reshape(Q, (1, sizeShape))[0]).tolist()

    A = np.ndarray(shape=np.add(shape,1), dtype=float)

    # Create an array of specified dimension
    grid = [range(i+1) for i in shape]
    zeroCoordinate = tuple( (0 for x in range(sizeShape)) )
    A[zeroCoordinate] = 1

    for coordinate in product(*grid): # iterate over A
        if coordinate != zeroCoordinate:
           print("------------------------------------")
           A[coordinate] = computeA(coordinate, A) # type: ?
           h = compute_h(coordinate,nrows, ncols,  P) # type: ?
           A[coordinate] *= 1/h


    return A[tuple(shape)]

# Test the function
P = np.array([[1.0,1.0] ,[1.0,1.0]], dtype=float)
Q = np.array([[1,2] ,[3,4]], dtype=int)

result = computeGeneralRecurrence1(Q, P)

print(result)
