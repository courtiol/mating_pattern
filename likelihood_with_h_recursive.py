import numpy as np
from math import sqrt
from math import factorial

import time

def memoize(f):
    cache = {}
    def worker(*args):
        if str(args) not in cache:
            cache[str(args)] = f(*args)
        return cache[str(args)]
    return worker

#recursive version

def h(coordinate, P):
    temp = sqrt(len(coordinate))
    q_shape = (temp , temp)
    mating_pattern = np.reshape(np.array(coordinate), q_shape)
    x, y = (mating_pattern.sum(axis=1), mating_pattern.sum(axis=0))
    return float(np.dot(x, np.dot(P, y)))

@memoize
def likelihoodR(coordinate, P):
    result = 0.0
    if all( [True if i==0 else False for i in coordinate] ):
        return 1
    for i in range(len(coordinate)):
        if coordinate[i]>0:#
            new_coordinate = [coordinate[j] if j != i else coordinate[j]-1 for j in range(len(coordinate))] # [0,...,-1,0,...]
            result += likelihoodR(new_coordinate, P)
    temp = h(coordinate, P)
    if temp > 0:
        result /= temp
    return float(result)

def likelihood(Q, P):
    x, y = (Q.sum(axis=1), Q.sum(axis=0))
    recurrenceFactor = np.prod([factorial(i) for i in np.concatenate((x, y))])*np.prod(P**Q)  # NB: x and y are same as in Q
    return likelihoodR(Q.flatten(), P)*recurrenceFactor

P = np.array([[0.5, 0.6, 0.8], [0.7, 0.8, 0.9], [0.5, 0.4, 0.2]], dtype=float)
Q = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=int)

ts = time.time()
res = likelihood(Q, P)
te = time.time()

print(res)
print("elapsed time "+str(te-ts))