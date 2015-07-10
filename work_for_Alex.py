import numpy as np

def computeA(coordinate, A):
    # input:
    #  - coordinate: a tuple, representing a given mating pattern
    #  - A: a multidimensional np.array
    # output:
    #  - result: an int, next term of the recurrence
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
    # input:
    #  - coordinate: a tuple
    #  - n: an integer defining the number of male types
    #  - m: an integer defining the number of female types
    #  - P: the preference matrix (np.array)
    # output:
    #  - h, the numerator of the probability that any mating occur during one time step
    mating_pattern = np.reshape(np.array(coordinate), (n, m))
    unity_x = [1 for i in range(m)]
    unity_y = [1 for i in range(n)]
    x = np.dot(mating_pattern, unity_x)  # vector of males (sum over cols)
    y = np.dot(np.transpose(mating_pattern), unity_y)  # vector of females (sum over rows)
    h = np.dot(y, np.dot(P, x))
    return h

'''
# example:
coordinate_test = (1, 2, 1, 1)
P_test = np.array([[0.2, 0.1], [0.4, 1.0]])
print(P_test)
print(compute_h(coordinate_test, 2, 2, P_test))
'''
