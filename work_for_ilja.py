def memoize(f):
    cache = {}
    def worker(*args):
        if args not in cache:
            print("calculating")
            cache[args] = f(*args)
        return cache[args]
    return worker

@memoize
cdef int square(int x):
    return x * x


if __name__ == '__main__':
    print(square(2))
    print(square(2))
    print(square(3))
