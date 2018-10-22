We use `complex_hook=True` in all cases except when explicitly specified.

## With clang 6.0

### branch master

```
## default 2d (no loop)
Mean +- std dev: 9.85 ms +- 0.03 ms
# default 2d (explicit loops)
Mean +- std dev: 5.56 ms +- 0.02 ms

# simd 2d (no loop)
Mean +- std dev: 7.47 ms +- 0.02 ms
# simd 2d (explicit loops)
Mean +- std dev: 5.72 ms +- 0.03 ms
```

### branch fix/complex-scalar-broadcast-perf

```
## default 2d (no loop)
Mean +- std dev: 6.28 ms +- 0.03 ms
# default 2d (explicit loops)
Mean +- std dev: 5.56 ms +- 0.02 ms

# simd 2d (no loop)
Mean +- std dev: 5.51 ms +- 0.06 ms
# simd 2d (explicit loops)
Mean +- std dev: 5.76 ms +- 0.11 ms
```

Same but without `complex_hook=True`:

```
## default 2d (no loop)
Mean +- std dev: 7.54 ms +- 0.02 ms
# default 2d (explicit loops)
Mean +- std dev: 5.62 ms +- 0.03 ms

# simd 2d (no loop)
Mean +- std dev: 6.55 ms +- 0.06 ms
# simd 2d (explicit loops)
Mean +- std dev: 5.61 ms +- 0.02 ms
```

## With gcc 6.3

### branch master

```
## default 2d (no loop)
Mean +- std dev: 10.8 ms +- 0.0 ms
# default 2d (explicit loops)
Mean +- std dev: 5.44 ms +- 0.03 ms

# simd 2d (no loop)
Mean +- std dev: 10.6 ms +- 0.0 ms
# simd 2d (explicit loops)
Mean +- std dev: 5.54 ms +- 0.03 ms

```

### branch fix/complex-scalar-broadcast-perf

```
## default 2d (no loop)
Mean +- std dev: 10.7 ms +- 0.0 ms
# default 2d (explicit loops)
Mean +- std dev: 5.44 ms +- 0.02 ms

# simd 2d (no loop)
Mean +- std dev: 10.6 ms +- 0.0 ms
# simd 2d (explicit loops)
Mean +- std dev: 5.54 ms +- 0.02 ms
```

# branch fix/more-complex-vectorize

### gcc 6.3

```
## default 2d (no loop)
Mean +- std dev: 10.6 ms +- 0.0 ms
# default 2d (explicit loops)
Mean +- std dev: 5.41 ms +- 0.01 ms

# simd 2d (no loop)
Mean +- std dev: 5.72 ms +- 0.01 ms
# simd 2d (explicit loops)
Mean +- std dev: 5.46 ms +- 0.01 ms
```

### clang 6.0

```
## default 2d (no loop)
Mean +- std dev: 5.56 ms +- 0.03 ms
# default 2d (explicit loops)
Mean +- std dev: 5.56 ms +- 0.02 ms

# simd 2d (no loop)
Mean +- std dev: 5.66 ms +- 0.01 ms
# simd 2d (explicit loops)
Mean +- std dev: 5.73 ms +- 0.04 ms
```