import numpy as np
import math


def computeA(coordinate, A):
    # this function computes the next term of the recurrence
    # coordinate is a tuple
    # A is a multidimensional np.array
    result = 0
    for i in range(len(coordinate)):
        if coordinate[i] > 0:
            # new tuple with same coordinates but -1 at position i:
            temp_coordinate = coordinate[:i]+(coordinate[i]-1,)+coordinate[(i+1):]
            result += A[temp_coordinate]
    return result

'''
# example in 2 dimensions:
A_test = np.array([[0, 2, 0], [5, 0, 0], [0, 0, 0]])
print(A_test)
coordinate_test = (1, 1)
print(computeA(coordinate_test, A_test)) # should be 7 (= 2 + 5)
'''

def compute_h(coordinate, n, m, P):
    # this function compute h, the probability that any mating occur during one time step
    # coordinate is an array
    # n is an integer defining the number of male types
    # m is an integer defining the number of female types
    # P is the preference matrix
    C = np.reshape(coordinate, (n, m))
    unity_x = [1 for i in range(m)]
    unity_y = [1 for i in range(n)]
    x = np.dot(C, unity_x)
    y = np.dot(np.transpose(C), unity_y)
    return np.dot(y, np.dot(P, x))

coordinate_test = np.array([1, 1])
P_test = np.array([[0.2, 0.1], [0.4, 1.0]])
print(P_test)
#print(compute_h(coordinate_test, 5, 5, P_test))




def computeGeneralPMatingpattern1(Q, P):
    unity_x = [1 for i in range(Q.ncols())] # ToDo: change to np.shape()
    unity_y = [1 for i in range(Q.nrows())]
    x = np.dot(np.array(Q),unity_x)
    y = np.dot(np.transpose(np.array(Q)), unity_y)

    prod = 1
    for xi in x:
        prod = prod*math.factorial(xi)
    for yi in y:
        prod = prod*math.factorial(yi)

    for i in range(Q.nrows()): # Or use np.flatten
        for j in range(Q.ncols()):
            prod = prod*(P[i][j]**Q[i][j] )
    print("prod "+str(prod))
    print("cP "+str(computeGeneralRecurrence1(Q, P)))

    return prod*computeGeneralRecurrence1(Q, P)