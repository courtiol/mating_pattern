import numpy as np
from itertools import product
import math
from work_for_Alex import compute_h, computeA

def computeGeneralRecurrence1(Q, P):
    """
    This method computes the recurrence A[Q] = 1/* ( A[Q-E11] + ... A[Q-E_mn])
    :param Q: multidimensional numpy.array
    :param P: multidimensional numpy.array
    :return: the recurrence term A[Q]
    """
    n = Q[0][0]
    nrows = np.shape(Q)[0]
    ncols = np.shape(Q)[1]
    sizeShape = ncols * nrows
    shape = (np.reshape(Q, (1, sizeShape))[0]).tolist()

    # Create an array of specified dimension
    dimension = nrows*ncols # We need an array of this size in each computation step
    zeroCoordinate = tuple([0 for x in range(dimension)])
    grid = [range(i+1) for i in shape]

    A = np.ndarray(shape=np.add(shape,1), dtype=int)
    A[zeroCoordinate] = 1

    for coordinate in product(*grid): # iterate over A
        if coordinate != zeroCoordinate:
           print("------------------------------------")
           A[coordinate] = computeA(coordinate, A)

           print("1: A["+str(coordinate)+"] "+str(A[coordinate]))
           print("1.5: A["+str(coordinate)+"] "+str(float(A[coordinate])))

           h = compute_h(coordinate,nrows, ncols,  P)
           A[coordinate] = 1/h*A[coordinate]

    return A[tuple(shape)]

# Test the function
#Q = Matrix([[2,3] ,[3,2]])
#P = Matrix([[1.0,1.0] ,[1.0,1.0]])
#print(computeGeneralPMatingpattern1(Q, P))

Q = np.array(np.mat('1 2 3; 3 4 3'))
P = np.array([[1.0,1.0] ,[1.0,1.0]])

print(computeGeneralRecurrence1(Q, P))
