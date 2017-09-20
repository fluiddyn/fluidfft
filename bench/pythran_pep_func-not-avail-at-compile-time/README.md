# Proposal for Pythran: type "python functions not available at compile time"

For the project fluidfft/fluidsim, I would need a new feature in Pythran : the
possibility to define exported functions that have as arguments "python
functions not available at compile time", i.e. "python functions to be called
by the python interpreter".

The goal is to be able to compile with Pythran functions using other functions
not known by Pythran, for example functions of a C extension (in practice often
produced with Cython).

As usually with Pythran, we have to restric ourself to simple things compared
to what we can do with Python, in pratice in this case, to functions that:

- do not change the "dtype" or the shape of their arguments (for example a list
  of integers has to remain a list of integer, also nothing exotic like
  https://stackoverflow.com/questions/4389517/in-place-type-conversion-of-a-numpy-array).
- "always return the same types", at least how they are used in the exported
  code.

These restrictions are actually not so strong and many functions respect them.

## Example

Here is a simple silly example of a module `mymod.py` which would have to be
pythranized:

```python

import numpy as np

# pythran export myfunc(
#     float[], int,
#     function_to_be_called_by_python_interpreter -> float64[]
#     function_to_be_called_by_python_interpreter -> complex128[])

def myfunc(a, n, func0, func1):

    c = a**2 + a**3 + 1

    tmp = np.empty(n)
    for i in range(n):
        tmp[i] = func0(i*c)

    ret = tmp**4 + 2*func1(a)

    return ret

```

In the Python code where it will be used, `func1` can be any Python functions
returning an array of complex128. In particular this function can use funky
compiled libraries (as for example in fluidfft), but Pythran does not have to
worry about it.

We see that even though Pythran can't do anything to speedup the execution of
the function `func1`, there are plenty of other opportunities of
optimization. We also see that pythran will not be lost in term of type
inference. It will be able to check that the statement `tmp[i] = func0(i*c)` is
correct and to infer that `ret` is an array of complex.

Even if this way of writting the function to be exported (with functions as
arguments) is not the most natural things to do when you do not use Pythran, I
see two advantages for Pythran:

- we stay in the normal framework of Pythran, one module with some functions,
  nothing actually complicated.

- it is very clear when you see the code that Pythran does not have to work on
  the actual code of the functions
  "function_to_be_called_by_python_interpreter". We clearly see that we do not
  have the code so the only thing that we can do is to work with the
  information that we have :

    * the types of the returned objects
    * the restrictions on the function_to_be_called_by_python_interpreter
    * how the function is used in the code

## Thoughts about the implementation

The type inference mechanisms of Pythran have to be able to use the information
on the return type of the functions
"function_to_be_called_by_python_interpreter". I do not see why it would be
really difficult.

In the C++ code, for each call of a function_to_be_called_by_python_interpreter:

1. get back the GIL (something like `PyEval_RestoreThread(_save);`)
2. create the necessary python objects (the arguments of the function) with
  something like `to_python(...);`
3. execute the Python function
4. check the types of the returned Python objects (and maybe also of the
   arguments after the function call)
5. convert the returned Python objects and the arguments to C++ objects with
   `from_python<pythonic::types::...>(...);`
6. release the GIL.

All these steps of course take time but nothing huge and except the actual call
of the Python function, every step already happens in Pythran extension (get
and release the GIL, conversions C++->Python and Python->C++ and type
checking).

## conclusions

For what I want to do with Pythran in fluidfft and fluidsim, this new feature
would be very very useful. I guess it can be useful for others.

In real code (not stackoverflow questions), it is very common in Python to use
compiled extensions and/or libraries with ctypes or ffi. When you use such
things in numerical computing, it is very convenient and natural to use these
functions in the core of the computational functions. And when you use Pythran,
you want to pythranize your functions without reorganizing too much our code.

When I think about it, it seems to me that there are no strong technical
barriers to implement this into Pythran. But of course, my knowledge about
Pythran internal is very very vague so...
