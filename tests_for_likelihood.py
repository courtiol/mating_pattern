import numpy as np
import likelihood_with_Fij as L1
import likelihood_with_Fij_for_c_perspective as L2

def translate(Q, P):
    Pflat = list(P.flatten())
    Qflat = list(Q.flatten())
    nrow = np.shape(Q)[0]
    ncol = np.shape(Q)[1]
    return Qflat, Pflat, nrow, ncol


Q = np.array([[50, 4], [30, 2]], dtype=int)
P = np.array([[1.0, 0.8], [0.5, 0.2]], dtype=float)
print(L1.likelihood(Q, P))
print(L2.likelihood(*translate(Q, P)))

Q = np.array([[1, 1], [1, 1]], dtype=int)
P = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)

print(L1.likelihood(Q, P, limit_zero=False))
print(L2.likelihood(*translate(Q, P), limit_zero=False))

print(L1.likelihood(Q, P, limit_zero=True))
print(L2.likelihood(*translate(Q, P), limit_zero=True))

Q = np.array([[5, 4, 4], [4, 2, 1], [1, 1, 1]], dtype=int)
P = np.array([[0.5, 0.6, 0.8], [0.7, 0.8, 0.9], [0.5, 0.4, 0.2]], dtype=float)
print(L1.likelihood(Q, P))
print(L2.likelihood(*translate(Q, P)))
