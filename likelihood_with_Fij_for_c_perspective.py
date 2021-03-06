def index_to_coordinate(index, dim):
    i = [1, ]*len(dim)
    for x in range(len(dim)):
        i[x] = int(index % (dim[x]+1))
        index -= i[x]
        index /= (dim[x]+1)
    return i
'''
# Test:
for i in range(0, 24):
    print(str(i)+" "+str(index_to_coordinate(i, dim=[1,2,3])))
'''


def coordinate_to_index(coordinate, max_coordinate):
    i = len(coordinate)-1
    value = 0
    while not i == -1:
        tmp = coordinate[i]
        for j in range(i):
            tmp *= (max_coordinate[j]+1)
        value += tmp
        i -= 1
    return value
'''
# Test:
coordinate_test = index_to_coordinate(25, dim=[2,5,2,3])
coordinate_to_index(coordinate_test, max_coordinate=[2,5,2,3])
'''

def xy_from_matrix(m, nrow, ncol):
    x = [0, ]*nrow
    y = [0, ]*ncol
    for j in range(0, nrow):
            for i in range(ncol):
                y[i] += m[ncol*j+i]
                x[j] += m[ncol*j+i]
    return x, y
'''
# Test:
xy_from_matrix([1,2,3,5,6,7], 2, 3)
'''

def likelihood(Q, P, nrow, ncol, limit_zero=False):
    coordinate_indexes = 1
    for i in range(nrow*ncol):
        coordinate_indexes *= (Q[i]+1)
    A = [1, ]*coordinate_indexes
    for coordinate_index in range(coordinate_indexes):
        coordinate = index_to_coordinate(coordinate_index, Q)
        if sum(coordinate) > 0:
            result = 0
            x, y = xy_from_matrix(coordinate, nrow, ncol)
            for i in range(nrow):
                for j in range(ncol):
                    index = ncol*i+j
                    if coordinate[index] > 0:
                        pos = coordinate.copy()
                        pos[index] -= 1
                        new_index = coordinate_to_index(pos, Q)
                        tmp = P[ncol*i+j]*x[i]*y[j]*A[new_index]
                        result += tmp
            h = 0
            for i in range(nrow):
                for j in range(ncol):
                    h += P[ncol*i+j]*x[i]*y[j]
            if h == 0 and result == 0:
                if limit_zero:  # to compute likelihood when pref tend to zero (to consider virtual types)
                    A[coordinate_to_index(coordinate, Q)] = 1
                else:
                    A[coordinate_to_index(coordinate, Q)] = 0  # to compute likelihood when pref really are zero
            else:
                A[coordinate_to_index(coordinate, Q)] = result/h
    return A[coordinate_to_index(Q, Q)]

if __name__ == '__main__':

    import time
    start = time.time()
    Q =[5, 4, 2, 3, 2, 1]
    P =[1.0, 0.8, 0.2, 0.5, 0.2, 0.7]
    print(likelihood(Q, P, nrow=2, ncol=3))
    stop = time.time()
    print("time = "+str(round(stop-start))+" sec")

