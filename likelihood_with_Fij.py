import numpy as np
from itertools import product
from math import factorial, log, exp, lgamma


def likelihood(Q, P, limit_zero=False):
    shape = Q.flatten()
    q_shape = np.shape(Q)
    A = np.ones(shape=np.add(shape, 1), dtype=float)  # create nd array initialised at one
    grid = [range(i+1) for i in shape]  # Create an array of specified dimension
    for coordinate in product(*grid):  # iterate over A
        if sum(coordinate) > 0:
            result = 0
            mating_pattern = np.reshape(np.array(coordinate), q_shape)
            x, y = (mating_pattern.sum(axis=1), mating_pattern.sum(axis=0)) # x = row sum, y = col sum
            for i in range(len(x)):
                for j in range(len(y)):
                    index = len(y)*i+j
                    if coordinate[index] > 0:
                        result += P[i, j]*x[i]*y[j]*A[coordinate[:index]+(coordinate[index]-1,)+coordinate[(index+1):]] # A[new tuple with same coordinates but -1 at position i] :
            h = np.dot(x, np.dot(P, y))
            if h == 0 and result == 0:
                if limit_zero: # to compute likelihood when pref tend to zero (to consider virtual types)
                    A[coordinate] = 1
                else:
                    A[coordinate] = 0 # to compute likelihood when pref really are zero
            else:
                A[coordinate] = result/h  # result is divided by h
    return A[tuple(shape)]

if __name__ == '__main__':
    #import yappi
    #yappi.start()
    #import statprof
    #statprof.start()
    #try:

    import time
    start = time.time()
    Q = np.array([[5, 4, 2], [3, 2, 1]], dtype=int)
    P = np.array([[1.0, 0.8, 0.2], [0.5, 0.2, 0.7]], dtype=float)
    #P = np.array([[1.0, 0.8, 0], [0.5, 0.2, 0]], dtype=float)
    #Q = np.array([[0, 1, 0], [1, 0, 2]], dtype=int)
    #P = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    #Q = np.array([[1, 1], [1, 1]], dtype=int)
    #P = np.array([[1.0, 1.0, 0.01], [1.0, 1.0, 0.01], [0.001, 0.001, 0]], dtype=float)
    #Q = np.array([[10, 10, 0], [10, 10, 0], [0, 0, 0]], dtype=int)
    #P = np.array([[0.5, 0.6, 0.8], [0.7, 0.8, 0.9], [0.5, 0.4, 0.2]], dtype=float)
    #Q = np.array([[20, 10, 10], [10, 20, 10], [1, 1, 1]], dtype=int)


    print(likelihood(Q, P))
    stop = time.time()
    print("time = "+str(round(stop-start))+" sec")

    '''
    finally:
        #yappi.get_func_stats().print_all()
        statprof.stop()
        statprof.display()
    '''