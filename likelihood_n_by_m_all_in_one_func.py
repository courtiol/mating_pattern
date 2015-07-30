import numpy as np
from itertools import product
from math import factorial
import time


def likelihood(Q, P):
    shape = Q.flatten()
    shape_range = range(len(shape))
    q_shape = np.shape(Q)
    A = np.ones(shape=np.add(shape, 1), dtype=float)  # create nd array initialised at one
    grid = [range(i+1) for i in shape]  # Create an array of specified dimension
    for coordinate in product(*grid):  # iterate over A
        if sum(coordinate) > 0:
            result = 0
            for i in shape_range:
                if coordinate[i] > 0:
                    result += A[coordinate[:i]+(coordinate[i]-1,)+coordinate[(i+1):]] # A[new tuple with same coordinates but -1 at position i] :
            mating_pattern = np.reshape(np.array(coordinate), q_shape)
            x, y = (mating_pattern.sum(axis=1), mating_pattern.sum(axis=0))
            A[coordinate] = result/np.dot(x, np.dot(P, y))  # result is divided by h
    recurrenceFactor = np.prod([factorial(i) for i in np.concatenate((x, y))])*np.prod(P**Q)  # NB: x and y are same as in Q
    return recurrenceFactor*A[tuple(shape)]


if __name__ == '__main__':
    #import yappi
    #yappi.start()
    #import statprof
    #statprof.start()
    try:
        ts = time.time()
        P = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=float)
        #P = np.array([[1.0, 1], [1, 1]], dtype=float)
        Q = np.array([[7, 3], [1, 2]], dtype=int)
        res = likelihood(Q, P)
        te = time.time()
        print(res)
        print("elapsed time "+str(te-ts))

    finally:
        pass
        #yappi.get_func_stats().print_all()
        #statprof.stop()
        #statprof.display()
