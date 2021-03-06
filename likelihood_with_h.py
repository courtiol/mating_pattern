import numpy as np
from itertools import product
from math import factorial, log, exp, lgamma


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
    #recurrenceFactor = np.prod([factorial(i) for i in np.concatenate((x, y))])*np.prod(P**Q)  # NB: x and y are same as in Q
    #return recurrenceFactor*A[tuple(shape)]
    log_recurrenceFactor = np.sum([lgamma(i+1) for i in np.concatenate((x, y))])+log(np.prod(P**Q)) # I do lgamma(i+1) as an equivalent of log(factorial(i))
    print("A = " +str(A[tuple(shape)]))
    return exp(log_recurrenceFactor+log(A[tuple(shape)]))

if __name__ == '__main__':
    #import yappi
    #yappi.start()
    #import statprof
    #statprof.start()
    #try:

    import time
    start = time.time()
    P = np.array([[1.0, 0.8, 0.6], [0.5, 0.3, 0.1]], dtype=float)
    Q = np.array([[5, 4, 3], [2, 1, 0]], dtype=int)
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