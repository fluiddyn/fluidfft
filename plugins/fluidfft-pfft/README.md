# Fluidfft plugin for parallel FFTs using PFFT

This plugin provides a method for parallel FFTs using PFFT:
`fft3d.mpi_with_pfft`.

## Environment variables

The default include path can be expanded with `CPATH` for GCC/Clang and
`INCLUDE` for MSVC.

The default library search path can be expanded with `LIBRARY_PATH` for
GCC/Clang and `LIB` for MSVC.

Alternatively, one could define `PFFT_DIR` or `PFFT_LIB_DIR` and
`PFFT_INCLUDE_DIR`.
