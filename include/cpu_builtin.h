/* Small utilities for inspecting the CPU using builtin functions */

#ifndef CPU_SIMD_H
#define CPU_SIMD_H

#if defined(__amd64__) || defined (_M_X64) || defined(__i386__) || defined(_M_IX86) || defined(_X86_)

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
#else

int simd_alignment(void){
    return 4;
}
#endif

#endif /* Header guard */
