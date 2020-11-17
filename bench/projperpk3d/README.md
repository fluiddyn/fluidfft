# Strange things!

## Same code but difference in performance!

See the files proj.py, proj1.py and proj2.py

pierre@pierre-KTH:~/Dev/fluidfft/bench/compare_projperpk3d$ make perfdefault
python -m pyperf timeit -s \
  'from bench import proj_default as func, arr_c, arr' 'func(arr_c, arr_c, arr_c, arr, arr, arr, arr)'
.....................
Mean +- std dev: 23.8 ms +- 1.3 ms


pierre@pierre-KTH:~/Dev/fluidfft/bench/compare_projperpk3d$ make perfdefault1
python -m pyperf timeit -s \
  'from bench import proj1_default as func, arr_c, arr' 'func(arr_c, arr_c, arr_c, arr, arr, arr, arr)'
.....................
Mean +- std dev: 26.6 ms +- 1.1 ms


pierre@pierre-KTH:~/Dev/fluidfft/bench/compare_projperpk3d$ make perfdefault2 
python -m pyperf timeit -s \
  'from bench import proj2_default as func, arr_c, arr' 'func(arr_c, arr_c, arr_c, arr, arr, arr, arr)'
.....................
Mean +- std dev: 23.9 ms +- 1.5 ms


## -march=native (slightly) slower than default

Small difference but it is reproducible...

pierre@pierre-KTH:~/Dev/fluidfft/bench/compare_projperpk3d$ make perfnative 
python -m pyperf timeit -s \
  'from bench import proj_native as func, arr_c, arr' 'func(arr_c, arr_c, arr_c, arr, arr, arr, arr)'
.....................
Mean +- std dev: 24.8 ms +- 1.3 ms

lscpu gives:

Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36
clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm
constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc
aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3
cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave
avx lahf_lm epb retpoline kaiser tpr_shadow vnmi flexpriority ept vpid xsaveopt
dtherm ida arat pln pts


## No effect of -DUSE_BOOST_SIMD


## Slow compared to Fortran (Mean Time = 19.320 ms)
