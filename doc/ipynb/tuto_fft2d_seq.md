---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tutorial FFT 2D sequential

In this tutorial, we present how to use fluidfft to perform 2d Fast Fourier Transform in sequential. You will see that it is really very simple.

+++

If you really want performance, first benchmark the different methods for an array shape representative of what you will use and in the machine where the computations will be done. We explain how to do this in the page [Benchmarking](../bench.md).

```{code-cell} ipython3
import numpy as np
from fluidfft import get_methods, import_fft_class
```

```{code-cell} ipython3
print(get_methods(ndim=2, sequential=True))
```

We import a class and instantiate it:

```{code-cell} ipython3
cls = import_fft_class('fft2d.with_pyfftw')
```

```{code-cell} ipython3
o = cls(16, 48)
```

Let's have a look at the attribute of this objects. They are actually all methods:

```{code-cell} ipython3
print('\n'.join([name for name in dir(o) if not name.startswith('__')]))
```

Let's run a test and benchmark the fft and ifft functions directly from C++.

```{code-cell} ipython3
print(o.run_tests())
```

```{code-cell} ipython3
t1, t2 = o.run_benchs()
print('t_fft = {} s; t_ifft = {} s'.format(t1, t2))
```

Let's understand how the data is stored:

```{code-cell} ipython3
print(o.get_is_transposed())
```

which means that for this class, in Fourier space, the data is not transposed...

Now we can get the non dimensional wavenumber in the first and second dimensions:

```{code-cell} ipython3
k0, k1 = o.get_k_adim_loc()
print('k0:', k0)
print('k1:', k1)
```

and check that the shapes of the array in one process are the same than in sequential (we are in sequential, there is only one process):

```{code-cell} ipython3
assert o.get_shapeX_loc() == o.get_shapeX_seq()
assert o.get_shapeK_loc() == o.get_shapeK_seq()
```

Now, let's compute fast Fourier transforms. We first initialize arrays:

```{code-cell} ipython3
a = np.ones(o.get_shapeX_loc())
a_fft = np.empty(o.get_shapeK_loc(), dtype=np.complex128)
```

If we do not have the array where to put the result we can do:

```{code-cell} ipython3
a_fft = o.fft(a)
```

If we already have the array where to put the result we can do:

```{code-cell} ipython3
o.fft_as_arg(a, a_fft)
```

And finally for the inverse Fourier transform:

```{code-cell} ipython3
a = o.ifft(a_fft)
```

```{code-cell} ipython3
o.ifft_as_arg(a_fft, a)
```

Let's mention the existence of the method ``ifft_as_arg_destroy``, which can be slightly faster than `ifft_as_arg` because it avoids one copy.

```{code-cell} ipython3
o.ifft_as_arg_destroy(a_fft, a)
```
