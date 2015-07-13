import numpy as np
from itertools import product
from math import factorial


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
    x = mating_pattern.sum(axis=0)
    y = mating_pattern.sum(axis=1)
    return np.dot(y, np.dot(P, x))  # h
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

    # factorial on all types, males (x=Q.sum(axis=0) and females (Q.sum(axis=1))
    prod = np.prod([factorial(i) for i in np.concatenate((Q.sum(axis=0), Q.sum(axis=1)))])
    prod *= np.prod(P**Q)
    return prod
'''
# example:
Q_test = np.array([[1, 2], [5, 2]])
P_test = np.array([[0.2, 0.1], [0.4, 1.0]])
print(computeRecurrenceFactor(Q_test, P_test))
'''


def computeGeneralPMatingpattern1(Q, P):
    """
    This method computes the recurrence A[Q] = 1/h* ( g_11*A[Q-E11] + ... *g_{m,m}*A[Q-E_mm])  // WRONG, now does full recurence
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
    return computeRecurrenceFactor(Q, P)*A[tuple(shape)]

if __name__ == '__main__':
    # Testcase:
    P = np.array([[1.0, 1.0, 0.0001], [1.0, 1.0, 0.0001], [0.0001, 0.0001, 0]], dtype=float)
    Q = np.array([[3, 1, 1], [1, 5, 1], [1, 1, 0]], dtype=int)
    result = computeGeneralPMatingpattern1(Q, P)
    print(result)
