import numpy as np
import random


def choose_random(v):  # Attempt to design a faster function (using stochastic acceptance)
    max_v = max(v)
    len_v = len(v)
    while 1:
        i = int(random.random()*len_v)
        if random.random() <= (v[i]/max_v):
            return i

'''
# test:
foo = np.array([0, 10, 20, 10, 60])
rec = np.array([choose_random(foo) for i in range(0, 100000)])
print([np.mean(rec == i) for i in range(0, len(foo))])
print("goal = "+str(foo/sum(foo)))
'''


def computeMatingPattern(x, y, P):
    if len(x) != len(P[:, 0]) | len(y) != len(P[0, :]):
        print("error: wrong dimensionality of P (or Q)")
        return 0
    if sum(x) > sum(y):
        Q = np.zeros((len(x), len(y)+1), dtype="int32")
    elif sum(x) < sum(y):
        Q = np.zeros((len(x)+1, len(y)), dtype="int32")
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
            print(i)
            Q[i, Q.shape[1]-1] = x[i]
    if sum(y) > 0:  # add unpaired females to Q
        for i in range(len(y)):
            Q[Q.shape[0]-1, i] = y[i]
    return Q

'''
# test:
males = np.array([1, 3])
females = np.array([5, 2])
Pref = np.array([[0.5, 0.5], [0.5, 0.5]])
computeMatingPattern(males, females, Pref)
'''


def countMatingPattern(x, y, P, number_simu):
    dict_Q = {}
    for i in range(number_simu):
        Q = computeMatingPattern(np.copy(x), np.copy(y), P)
        if str(Q) not in dict_Q.keys():
            dict_Q[str(Q)] = 1
        else:
            dict_Q[str(Q)] += 1
    return dict_Q
'''
# test:
males = np.array([10, 3])
females = np.array([5, 2])
Pref = np.array([[0.5, 0.5], [0.5, 0.5]])
countMatingPattern(males, females, Pref, 10)
'''


def freqMatingPattern(Q, P, number_simu):
    if np.shape(Q)[0] != np.shape(Q)[1] | np.shape(P)[0] != np.shape(P)[1]:
        print("error: np arrays for Q and P must be square matrices")
        return 0
    x = Q.sum(axis=1)
    y = Q.sum(axis=0)
    dict_Q = countMatingPattern(x, y, P, number_simu)
    print(dict_Q)
    if str(Q) not in dict_Q.keys():
        print("Warning: kys for Q not found")
        return 0
    nb = dict_Q[str(Q)]
    return nb/number_simu


if __name__ == '__main__':
    # import yappi
    # yappi.start()
    # import statprof
    # statprof.start()
    # try:

    import time
    start = time.time()
    P = np.array([[1.0, 0.8], [0.5, 0.1]], dtype=float)
    Q = np.array([[5, 4], [3, 2]], dtype=int)
    print(Q)
    #P = np.array([[1.0, 1.0, 0.01], [1.0, 1.0, 0.01], [0.001, 0.001, 0]], dtype=float)
    #Q = np.array([[10, 10, 0], [10, 10, 0], [0, 0, 0]], dtype=int)

    print(freqMatingPattern(Q, P, 10000))
    stop = time.time()
    print("time = "+str(round(stop-start))+" sec")

    # finally:
    #     yappi.get_func_stats().print_all()
    #     statprof.stop()
    #     statprof.display()