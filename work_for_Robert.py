import numpy as np
from itertools import product
from likelihood_n_by_m import compute_h, computeA
from math import factorial


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
    zeroCoordinate = tuple( (0 for _ in range(sizeShape)) )
    A[zeroCoordinate] = 1

    for coordinate in product(*grid): # iterate over A
        if coordinate != zeroCoordinate:
           h = compute_h(coordinate,nrows, ncols,  P) # type: ?
           A[coordinate] = computeA(coordinate, A)/h # type: ?


    return A[tuple(shape)]


def computeGeneralPMatingpattern1(Q, P):
    unity_x = [1 for i in range(Q.ncols())] # ToDo: change to np.shape()
    unity_y = [1 for i in range(Q.nrows())]
    x = np.dot(np.array(Q),unity_x)
    y = np.dot(np.transpose(np.array(Q)), unity_y)

    prod = 1
    for xi in x:
        prod = prod*factorial(xi)
    for yi in y:
        prod = prod*factorial(yi)

    for i in range(Q.nrows()): # Or use np.flatten
        for j in range(Q.ncols()):
            prod = prod*(P[i][j]**Q[i][j] )
    print("prod "+str(prod))
    print("cP "+str(computeGeneralRecurrence1(Q, P)))

    return prod*computeGeneralRecurrence1(Q, P)

# Test the function
P = np.array([[1.0,1.0] ,[1.0,1.0]], dtype=float)
Q = np.array([[2,1] ,[1,2]], dtype=int)

result = computeGeneralRecurrence1(Q, P)

print(result)
