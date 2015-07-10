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