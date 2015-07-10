import numpy as np
from itertools import product
import math

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
    
def computeGeneralRecurrence1(Q, P):
    n = Q[0][0]
    sizeShape = Q.ncols() * Q.nrows()
    shape = (np.reshape(Q, (1, sizeShape))[0]).tolist()
    
    # Create an array of specified dimension
    dimension = Q.nrows()*Q.ncols() # We need an array of this size in each computation step
    zeroCoordinate = tuple([0 for x in range(dimension)])
    grid = [range(i+1) for i in shape]
    
    A = np.ndarray(shape=np.add(shape,1), dtype=int)    
    A[zeroCoordinate] = 1
    
    for coordinate in product(*grid): # iterate over A
        if coordinate != zeroCoordinate:
           print("------------------------------------")
           A[coordinate] = computeA(coordinate, A)
           
           print("1: A["+str(coordinate)+"] "+str(A[coordinate]))
           print("1.5: A["+str(coordinate)+"] "+str(float(A[coordinate])))
           
           h = compute_h(coordinate,Q.nrows(), Q.ncols(),  P)
           A[coordinate] = float(1)/float(h)*float(A[coordinate]) # remove float in python
           
           print("1/h["+str(coordinate)+"] "+str( float(1)/float(h) ) )
           print("2: A["+str(coordinate)+"] "+str(A[coordinate]))
    
    return A[tuple(shape)]
           
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

