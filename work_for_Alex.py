def computeA(coordinate, A):
    result = 0
    for i in range(len(coordinate)):
        if coordinate[i] > 0:
            # create coordinate coordinate-e_i
            tempCoordinate = np.copy(coordinate)
            tempCoordinate[i] = tempCoordinate[i]-1
            tempCoordinate = tuple(tempCoordinate)
            result = result + A[tempCoordinate]
    return result

def compute_h(coordinate, n, m,  P):
    C = np.reshape(coordinate, (n, m))
    unity_x = [1 for i in range(m)]
    unity_y = [1 for i in range(n)]
    x = np.dot(C,unity_x)
    y = np.dot(np.transpose(C), unity_y)
    return np.dot(y,np.dot(P,x))

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