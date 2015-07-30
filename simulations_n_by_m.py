import numpy as np
import random

def choose_random(v):
    v_pct = [i/sum(v) for i in v]
    choice = np.random.multinomial(1, v_pct)
    return sum(choice*list(range(0, len(v))))

def computeMatingPattern(x, y, P):
    if len(x) != len(P[:, 0]) | len(y) != len(P[0, :]):
        print("error: wrong dimensionality of P (or Q)")
        return 0
    if sum(x) != sum(y):
        Q = np.zeros((len(x)+1, len(y)+1), dtype="int32")
    else:
        Q = np.zeros((len(x), len(y)), dtype="int32")
    while sum(x) > 0 and sum(y) > 0:
        cx = choose_random(x)
        cy = choose_random(y)
        if random.random() <= P[cx, cy]:
            x[cx] -= 1
            y[cy] -= 1
            Q[cx, cy] += 1
    if sum(x) > 0:  # add unpaired males to Q
        for i in range(len(x)):
            Q[i, len(y)] = x[i]
    if sum(y) > 0:  # add unpaired females to Q
        for i in range(len(y)):
            Q[len(x), i] = y[i]
    return Q

def countMatingPattern(x, y, P, number_simu):
    dict_Q = {}
    for i in range(number_simu):
        Q = computeMatingPattern(np.copy(x), np.copy(y), P)
        if str(Q) not in dict_Q.keys():
            dict_Q[str(Q)] = 1
        else:
            dict_Q[str(Q)] += 1
    return dict_Q

def freqMatingPattern(Q, P, number_simu):
    if np.shape(Q)[0] != np.shape(Q)[1] | np.shape(P)[0] != np.shape(P)[1]:
        print("error: np arrays for Q and P must be square matrices")
        return 0
    x = Q.sum(axis=1)
    y = Q.sum(axis=0)
    dict_Q = countMatingPattern(x, y, P, number_simu)
    if str(Q) not in dict_Q.keys():
        print("warning: keys for Q"+str(Q)+" not found")
        return 0
    nb = dict_Q[str(Q)]
    return nb/number_simu


if __name__ == '__main__':
    import yappi
    yappi.start()
    try:
        '''
        # Test of computeMatingPattern
        x = np.array([1, 2])
        y = np.array([2, 2])
        P = np.array([[0.5, 1.0], [0.7, 0.1]], dtype=float)
        test = computeMatingPattern(x, y, P)
        print(test)
        '''

        # Frequency of a given mating pattern in the universe

        P = np.array([[1.0, 1.0, 0.001], [1.0, 1.0, 0.0001], [0.001, 0.001, 0]], dtype=float)
        Q = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=int)
        print(freqMatingPattern(Q, P, 5))

    finally:
        yappi.get_func_stats().print_all()
