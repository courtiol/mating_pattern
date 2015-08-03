import numpy as np
from itertools import product
from math import factorial, log, exp, lgamma


def index_to_coordinate(index, dim):
    i = np.ones(shape=len(dim), dtype=int)
    for x in range(0, len(dim)):
        i[x] = index % (dim[x]+1)
        index -= i[x]
        index /= (dim[x]+1)
    return i
'''
# Test:
for i in range(0, 24):
    print(str(i)+" "+str(index_to_coordinate(i, dim=np.array([1,2,3], dtype=int))))
'''


def coordinate_to_index(coordinate, max_coordinate):
    i = len(coordinate)-1
    value = 0
    while not i == -1:
        tmp = coordinate[i]
        for j in range(0, i):
            tmp *= (max_coordinate[j]+1)
        value += tmp
        i -= 1
    return value

'''
# Test:
coordinate_test = index_to_coordinate(25, dim=np.array([2,5,2,3], dtype=int))
coordinate_to_index(coordinate_test, max_coordinate=np.array([2,5,2,3], dtype=int))
'''

def xy_from_matrix(m, nrow, ncol):
    x = np.zeros(shape=nrow, dtype=int)
    y = np.zeros(shape=ncol, dtype=int)
    for j in range(0, len(x)):
            for i in range(0, len(y)):
                y[i] += m[len(y)*j+i]
                x[j] += m[len(y)*j+i]
    return(x, y)

# Test:
#xy_from_matrix(np.array([1,2,3,5,6,7], dtype=int), 2, 3)


def likelihood(Q, P, limit_zero=False):
    shape = Q.flatten()
    q_shape = np.shape(Q)
    A = np.ones(shape=np.add(shape, 1), dtype=float)  # create nd array initialised at one
    coordinate_indexes = 1
    for i in range(len(shape)):
        coordinate_indexes *= (shape[i]+1)
    for coordinate_index in range(0, coordinate_indexes):  # iterate over A
        coordinate = tuple(index_to_coordinate(coordinate_index, shape))
        if sum(coordinate) > 0:
            result = 0
            x, y = xy_from_matrix(np.array(coordinate, dtype=int), q_shape[0], q_shape[1])
            for i in range(len(x)):
                for j in range(len(y)):
                    index = len(y)*i+j
                    if coordinate[index] > 0:
                        result += P[i, j]*x[i]*y[j]*A[coordinate[:index]+(coordinate[index]-1,)+coordinate[(index+1):]] # A[new tuple with same coordinates but -1 at position i] :
            if np.dot(x, np.dot(P, y)) == 0 and result == 0:
                if limit_zero: # to compute likelihood when pref tend to zero (to consider virtual types)
                    A[coordinate] = 1
                else:
                    A[coordinate] = 0 # to compute likelihood when pref really are zero
            else:
                A[coordinate] = result/np.dot(x, np.dot(P, y))  # result is divided by h
    return A[tuple(shape)]

if __name__ == '__main__':
    #import yappi
    #yappi.start()
    #import statprof
    #statprof.start()
    #try:

    import time
    start = time.time()
    P = np.array([[1.0, 0.8], [0.5, 0.2]], dtype=float)
    Q = np.array([[50, 4], [30, 2]], dtype=int)
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