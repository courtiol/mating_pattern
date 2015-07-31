import numpy as np
import likelihood_with_h
import likelihood_with_Fij
import  likelihood_with_h_recursive
import simulations_n_by_m
from functools import partial

test_data = []

P = np.array([[0.5, 0.7], [0.2, 0.8]], dtype=float)
Q = np.array([[6, 3], [1, 2]], dtype=int)
test_data.append((Q,P))

P = np.array([[1.0, 0.6], [0.7, 0.87]], dtype=float)
Q = np.array([[4, 3], [2, 2]], dtype=int)
test_data.append((Q,P))

P = np.array([[1.0, 0.6, 1], [0.7, 0.87, 0.7], [0.75, 0.8, 0.65]], dtype=float)
Q = np.array([[1, 2, 1], [2, 2, 1], [1, 1, 1]], dtype=int)
test_data.append((Q,P))

print("Start testing")
all_results = []
n = 10000

functions = []
functions.append(partial(simulations_n_by_m.freqMatingPattern, number_simu=1000))
functions.append(likelihood_with_h.likelihood)
functions.append(likelihood_with_Fij.likelihood)

for Q,P in test_data:
    result = []
    for func in functions:
        result.append(func(Q,P))
    all_results.append((Q,P,result))

print(all_results)