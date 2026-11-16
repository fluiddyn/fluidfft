/* Small utilities for inspecting the CPU using builtin functions */

#ifndef CPU_SIMD_H
#define CPU_SIMD_H

#if defined(__amd64__) || defined (_M_X64) || defined(__i386__) || defined(_M_IX86) || defined(_X86_)

#if defined(_MSC_VER)
/* MSVC and clang-cl: __builtin_cpu_init / __builtin_cpu_supports are lowered
 * to calls to __cpu_indicator_init / __cpu_model (libgcc / compiler-rt
 * builtins), which are not linked on Windows (lld-link: undefined symbol).
 * Use the cpuid intrinsic instead. */
#include <intrin.h>

#define FLUIDFFT_AVX_BIT 28 /* ecx, leaf 1 */
#define FLUIDFFT_SSE_BIT 25 /* edx, leaf 1 */

static int simd_alignment(void) {
  int cpuinfo[4];
  __cpuid(cpuinfo, 1);
  if (cpuinfo[2] & (1 << FLUIDFFT_AVX_BIT))
    return 32;
  else if (cpuinfo[3] & (1 << FLUIDFFT_SSE_BIT))
    return 16;
  else
    return 4;
}

#else

/* Returns the byte alignment for optimum simd operations */
int simd_alignment(void) {
#ifndef __APPLE__
  __builtin_cpu_init();
#endif
  if(
      __builtin_cpu_supports("avx")
      || __builtin_cpu_supports("avx2")
#if (__GNUC__ > 4)
      || __builtin_cpu_supports("avx512f")
#endif
  )
    return 32;
  else if(
      __builtin_cpu_supports("sse") ||
      __builtin_cpu_supports("sse2") ||
      __builtin_cpu_supports("sse3") ||
      __builtin_cpu_supports("ssse3") ||
      __builtin_cpu_supports("sse4.1") ||
      __builtin_cpu_supports("sse4.2")
  )
    return 16;
  else
    return 4;
}
#endif /* _MSC_VER */
#else

int simd_alignment(void){
    return 4;
}
#endif

#endif /* Header guard */
