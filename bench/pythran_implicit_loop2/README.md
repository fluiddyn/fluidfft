# benchmark Pythran on array operations

I investigate whether it would be possible to speedup some array operations with
Pythran.

## Methods

We benchmark simple functions involving only array operations (see mymod.py).

The 2d and 3d cases are similar so we can focus here only on the 3d case.

- myfunc_ret: the formula is expressed à la Numpy. The result is returned (a new
  array is allocated to store it)

- myfunc: the formula is expressed à la Numpy. The result is put in the first
  argument.  No memory allocation is necessary.

- myfunc_loops3d: version with 3 nested explicit loops (specialized for 3d).

- myfunc_loops_reshape: version with one reshape((-1,)) and one explicit
  loop. This version is not specialized for the 3d case and also works in 2d.

We use the functions with float and complex and the results are surprisingly
different.

Note that the complex and float arrays have the same size in memory, i.e. there
are twice more elements in the float64 arrays than in the complex128 arrays.

These results are obtained with clang and with `complex_hook = True`. Without
complex_hook, Pythran is slow with complex.

## Results

### 3d, float, no loop, return, py
Mean +- std dev: 357 ms +- 7 ms

### 3d, float, no loop, inplace, py
Mean +- std dev: 347 ms +- 14 ms

### 3d, float, no loop, return
Mean +- std dev: 25.3 ms +- 0.9 ms

### 3d, float, no loop, inplace
Mean +- std dev: 29.3 ms +- 2.4 ms

### 3d, float, explicit loops reshape, inplace
Mean +- std dev: 28.9 ms +- 1.3 ms

### 3d, float, explicit loops, inplace
Mean +- std dev: 29.9 ms +- 1.7 ms

- With Numpy, the "return" version is nearly as fast as the "inplace" version.

- Pythran is very good to speedup (~25 times faster) the code "à la Numpy with
  return" (without explicit loop and with memory allocation).

- All other versions (without allocation, without and with explicit loops) are as
  efficient (=> for float64, it is not necessary to write the loops explicitly)
  but slower than the version with allocation (!).

### 3d, complex, no loop, return, py
Mean +- std dev: 95.3 ms +- 16.7 ms

### 3d, complex, no loop, inplace, py
Mean +- std dev: 102 ms +- 16 ms

### 3d, complex, no loop, return
Mean +- std dev: 38.0 ms +- 1.7 ms

### 3d, complex, no loop, inplace
Mean +- std dev: 32.5 ms +- 8.6 ms

### 3d, complex, explicit loops reshape, inplace
Mean +- std dev: 18.1 ms +- 2.9 ms

### 3d, complex, explicit loops, inplace
Mean +- std dev: 17.2 ms +- 0.9 ms

- Numpy is much better with complex than with float (~4 times faster)! It is
  surprizing because there are the same number of float64 (if one considers that a
  complex128 is 2 float64) to be processed in the 2 cases.

- With Numpy, the inplace version is slightly slower than the "return" version.

- The speedup by Pythran of the code "à la Numpy" is much less impressive ("only"
  2 times faster for the "return" version and 3.6 times faster for the inplace
  version).

- We get faster function by writing the loop explicitly.

- The version with reshape and one explicit loop (which is not 3d specialized) is
  approximately as fast as the version with three nested loops.
