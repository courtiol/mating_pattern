import numpy as np
from itertools import product
from math import factorial


def computeGeneralPMatingpattern(Q, P):
    shape = Q.flatten()
    A = np.ones(shape=np.add(shape, 1), dtype=float)  # create nd array initialised at one
    grid = [range(i+1) for i in shape]  # Create an array of specified dimension
    for coordinate in product(*grid):  # iterate over A
        if sum(coordinate) > 0:
            result = 0
            for i in range(len(coordinate)):
                if coordinate[i] > 0:
                    result += A[coordinate[:i]+(coordinate[i]-1,)+coordinate[(i+1):]] # A[new tuple with same coordinates but -1 at position i] :
            mating_pattern = np.reshape(np.array(coordinate), np.shape(Q))
            x, y = (mating_pattern.sum(axis=0), mating_pattern.sum(axis=1))
            A[coordinate] = result/np.dot(y, np.dot(P, x))  # result is divided by h
    recurrenceFactor = np.prod([factorial(i) for i in np.concatenate((x, y))])*np.prod(P**Q)  # NB: x and y are same as in Q
    return recurrenceFactor*A[tuple(shape)]


if __name__ == '__main__':
    import yappi
    yappi.start()
    try:
        # Testcase:
        P = np.array([[0.1, 0.001], [0.2, 0.001]], dtype=float)
        Q = np.array([[10, 20], [20, 10]], dtype=int)
        print(computeGeneralPMatingpattern(Q, P))

    finally:
        yappi.get_func_stats().print_all()
