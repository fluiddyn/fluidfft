# fluidfft-hipfft

Fluidfft plugin using HIP/ROCm FFT library (hipFFT) for GPU-accelerated FFT operations.

This plugin provides GPU acceleration for 2D FFT operations using AMD's hipFFT library,
which is part of the ROCm software stack. It's compatible with AMD GPUs and can also
work with NVIDIA GPUs through HIP's CUDA backend.

## Installation

This plugin requires:
- ROCm (AMD) or HIP with CUDA backend (NVIDIA)
- hipFFT library

## Features

- 2D real-to-complex and complex-to-real FFT transforms
- GPU-accelerated computations
- Compatible with AMD ROCm GPUs

## License

See LICENSE file.
