import numpy as np
from itertools import product
from work_for_Alex import compute_h, computeA, computeRecurrenceFactor


def computeGeneralRecurrence1(Q, P):
    """
    This method computes the recurrence A[Q] = 1/h* ( A[Q-E11] + ... A[Q-E_mn])
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
    zeroCoordinate = tuple( (0 for _ in range(sizeShape)) )
    A[zeroCoordinate] = 1

    for coordinate in product(*grid): # iterate over A
        if coordinate != zeroCoordinate:
           h = compute_h(coordinate,nrows, ncols,  P) # type: ?
           A[coordinate] = computeA(coordinate, A)/h # type: ?


    return A[tuple(shape)]

def computeGeneralPMatingpattern1(Q, P):
    """
    This method computes the recurrence A[Q] =  f_11*A[Q-E11] + ... f_mn*A[Q-E_mn])
    :param Q: multidimensional numpy.array of type <class 'numpy.ndarray'> with coefficients of <class 'numpy.int64'>
    :param P: multidimensional numpy.array of type <class 'numpy.ndarray'> with coefficients of <class 'numpy.float64'>
    :return: the recurrence term A[Q] of type <class 'numpy.float64'>
    """
    prod = computeRecurrenceFactor(Q, P)
    return prod*computeGeneralRecurrence1(Q, P)


# Testcase:
P = np.array([[0.5,1.0] ,[0.7,0.1]], dtype=float)
Q = np.array([[1,2] ,[2,1]], dtype=int)
result = computeGeneralPMatingpattern1(Q, P)

print(result)
