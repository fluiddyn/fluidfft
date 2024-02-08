# Fluidfft plugin for parallel FFTs using P3DFFT

This plugin provides a method for parallel FFTs using P3DFFT:
`fft3d.mpi_with_p3dfft`.

## Environment variables

The default include path can be expanded with `CPATH` for GCC/Clang and
`INCLUDE` for MSVC.

The default library search path can be expanded with `LIBRARY_PATH` for
GCC/Clang and `LIB` for MSVC.

Alternatively, one could define `P3DFFT_DIR` or `P3DFFT_LIB_DIR` and
`P3DFFT_INCLUDE_DIR`.
