# Issue #14

https://bitbucket.org/fluiddyn/fluidfft/issues/14/segmentation-fault-with-fft2d

1. investigate_segfault.py: the main bug.

2. investigate_small.py: show (sometimes) a problem of memory conservation. I
   guess it is related to the main bug...

3. cpp directory: Pure c++ code leading (sometimes) to segfault. Just use `make`
   (many times if it works).
