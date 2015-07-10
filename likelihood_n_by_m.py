import numpy as np
from itertools import product
from math import factorial


def recover_types(Q):
    # input:
    #  - Q: np.array of a mating pattern
    # output:
    #  - x, y: vectors of males and females
    unity_x = [1, ] * np.shape(Q)[0]
    unity_y = [1, ] * np.shape(Q)[1]
    x = np.dot(Q, unity_x)
    y = np.dot(np.transpose(Q), unity_y)
    return x, y
'''
# example
Q_test = np.array([[1, 2], [5, 2]])
print(recover_types(Q_test))
'''


def computeA(coordinate, A):
    # input:
    #  - coordinate: a tuple, representing a given mating pattern
    #  - A: a multidimensional np.array
    # output:
    #  - result: an int, next term of the recurrence

    result = 0
    for i in range(len(coordinate)):
        if coordinate[i] > 0:
            # new tuple with same coordinates but -1 at position i:
            temp_coordinate = coordinate[:i]+(coordinate[i]-1,)+coordinate[(i+1):]
            result += A[temp_coordinate]
    return result
'''
# example in 2 dimensions:
A_test = np.array([[0, 2, 0], [5, 0, 0], [0, 0, 0]])
print(A_test)
coordinate_test = (1, 1)
print(computeA(coordinate_test, A_test)) # should be 7 (= 2 + 5)
'''


def compute_h(coordinate, n, m, P):
    # input:
    #  - coordinate: a tuple
    #  - n: an integer defining the number of male types
    #  - m: an integer defining the number of female types
    #  - P: the preference matrix (np.array)
    # output:
    #  - h, the numerator of the probability that any mating occur during one time step

    mating_pattern = np.reshape(np.array(coordinate), (n, m))
    x, y = recover_types(mating_pattern)
    h = np.dot(y, np.dot(P, x))
    return h
'''
# example:
coordinate_test = (1, 2, 1, 1)
P_test = np.array([[0.2, 0.1], [0.4, 1.0]])
print(P_test)
print(compute_h(coordinate_test, 2, 2, P_test))
'''


def computeRecurrenceFactor(Q, P):
    # input:
    #  - Q: np.array of a mating pattern
    #  - P: np.array of a mating preferences
    # output:
    #  - prod: the product needed for recurrence computation
    x, y = recover_types(Q)
    prod = np.prod([factorial(i) for i in np.concatenate((x, y))])
    prod *= np.prod(P.flatten()**Q.flatten())
    return prod
'''
# example:
Q_test = np.array([[1, 2], [5, 2]])
P_test = np.array([[0.2, 0.1], [0.4, 1.0]])
print(computeRecurrenceFactor(Q_test, P_test))
'''


def computeGeneralRecurrence1(Q, P):
    """
    This method computes the recurrence A[Q] = 1/h* ( A[Q-E11] + ... A[Q-E_mn])
    :param Q: multidimensional numpy.array of type <class 'numpy.ndarray'> with coefficients of <class 'numpy.int64'>
    :param P: multidimensional numpy.array of type <class 'numpy.ndarray'> with coefficients of <class 'numpy.float64'>
    :return: the recurrence term A[Q] of type <class 'numpy.float64'>
    """
    n = np.shape(Q)[0]
    m = np.shape(Q)[1]
    shape = Q.flatten()
    A = np.ndarray(shape=np.add(shape, 1), dtype=float)
    grid = [range(i+1) for i in shape]  # Create an array of specified dimension
    zeroCoordinate = (0,)*n*m  # nice way to create repeated values in a tuple!
    A[zeroCoordinate] = 1
    for coordinate in product(*grid):  # iterate over A
        if coordinate != zeroCoordinate:
           h = compute_h(coordinate, n, m, P)  # type: ?
           A[coordinate] = computeA(coordinate, A)/h  # type: ?

    return A[tuple(shape)]


def computeGeneralPMatingpattern1(Q, P):
    """
    This method computes the recurrence A[Q] =  f_11*A[Q-E11] + ... f_mn*A[Q-E_mn])
    :param Q: multidimensional numpy.array of type <class 'numpy.ndarray'> with coefficients of <class 'numpy.int64'>
    :param P: multidimensional numpy.array of type <class 'numpy.ndarray'> with coefficients of <class 'numpy.float64'>
    :return: the recurrence term A[Q] of type <class 'numpy.float64'>
    """
    prod = computeRecurrenceFactor(Q, P)*computeGeneralRecurrence1(Q, P)
    return prod


# Testcase:
P = np.array([[0.5,1.0] ,[0.7,0.1]], dtype=float)
Q = np.array([[1,2] ,[2,1]], dtype=int)
result = computeGeneralPMatingpattern1(Q, P)
print(result)
