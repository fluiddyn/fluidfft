
# pythran export myfunc(float64[][]):

def myfunc(a):
    return (a**2 + a**3 + 2) / 5.


# pythran export myfunc2(float64[][])

def myfunc2(a):
    return 2*a, 1
