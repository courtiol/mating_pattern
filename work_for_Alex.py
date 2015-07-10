import numpy as np
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


def recover_types(Q):
    # input:
    #  - Q: np.array of a mating pattern
    # output:
    #  - x, y: vectors of males and females
    nrows = np.shape(Q)[0]
    ncols = np.shape(Q)[1]
    unity_x = [1 for i in range(ncols)]  # ToDo: change to np.shape()
    unity_y = [1 for i in range(nrows)]
    x = np.dot(np.array(Q), unity_x)
    y = np.dot(np.transpose(np.array(Q)), unity_y)
    return x, y

'''
# example
Q_test = np.array([[1, 2], [5, 2]])
print(recover_types(Q_test))
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
    x, y = recover_types(Q)
    prod = 1
    for xi in x:
        prod *= factorial(xi)
    ## in progress
    print([x]+[y])
    np.prod([factorial(i) for i in x])

    for yi in y:
        prod *= factorial(yi)


    for i in range(len(x)): # Or use np.flatten
        for j in range(len(y)):
            prod = prod*(P[i][j]**Q[i][j])
    return prod

Q_test = np.array([[1, 2], [5, 2]])
P_test = np.array([[0.2, 0.1], [0.4, 1.0]])
print(computeRecurrenceFactor(Q_test, P_test))
