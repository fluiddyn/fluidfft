SINGLE_PREC:=yes
OMP:=no
CUDA:=no
PFFT:=no
P3DFFT:=no

LD:=mpic++
CC:=mpic++
FLAGS:= -g -O3 -Wall
LDFLAGS:=
LIBS:= -ldl -lhwloc -lm -lgfortran

LIB_DIR:=$(LDEV_PATH) -L$(FFTW3_LIB_DIR) -L/scratch/cnt0022/egi2153/SHARED/opt/p3dfft/2.7.5/lib -L/scratch/cnt0022/egi2153/SHARED/opt/pfft/1.0.6/lib
#INC_DIR:=-I. -I../base $(IDEV_PATH) -I$(FFTW3_INC_DIR) 
#CINES/occigen version
INC_DIR:=-I. -I../base $(IDEV_PATH) -I$(FFTW3_INC_DIR) -I/panfs/panasas/softs/intel/composer_xe_2015.3.187/compiler/include -I/scratch/cnt0022/egi2153/SHARED/opt/p3dfft/2.7.5/include -I/scratch/cnt0022/egi2153/SHARED/opt/pfft/1.0.6/include


SOURCES_CPP := $(wildcard base*.cpp)
SOURCES_CPP += fft3d_with_fftw3d.cpp fft3dmpi_with_fftw1d.cpp fft3dmpi_with_fftwmpi3d.cpp test_bench.cpp


ifeq "$(CUDA)" "yes"
SOURCES_CU := $(wildcard *.cu)

FLAGS+=-DCUDA
FLAGS_CU:=-m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
LD:=nvcc
LIBS+=-lcufft -lmpi_cxx -lmpi -lmpi_f77 $(LDEV_PATH) #-laf
INC_DIR_CU:=-I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi -I$(CPATH)
# INC_DIR+=-I/opt/cuda/NVIDIA_CUDA-6.0_Samples/common/inc/
endif

ifeq "$(OMP)" "yes"
  FLAGS+=-DOMP
  ifeq "$(CUDA)" "yes"
    FLAGS+= -fopenmp
    LDFLAGS+=-Xcompiler -fopenmp
  else
    FLAGS+= -fopenmp
    LDFLAGS+=-fopenmp
  endif
endif

ifeq "$(SINGLE_PREC)" "yes"
  FLAGS+=-DSINGLE_PREC
  LIBS+= -lfftw3f -lfftw3f_mpi -lfftw3f_omp
  FLAGS_CU+=-DSINGLE_PREC
  ifeq "$(PFFT)" "yes"
    LIBS+= -lpfftf
  endif
  ifeq "$(P3DFFT)" "yes"
    LIBS+= -lp3dfft_sp
  endif
else
  LIBS+= -lfftw3 -lfftw3_mpi -lfftw3_omp
  ifeq "$(PFFT)" "yes"
    LIBS+= -lpfft
  endif
  ifeq "$(P3DFFT)" "yes"
    LIBS+= -lp3dfft
  endif
endif

ifeq "$(PFFT)" "yes"
  FLAGS+= -DPFFT
  SOURCES_CPP+=fft3dmpi_with_pfft.cpp
endif

ifeq "$(P3DFFT)" "yes"
  FLAGS+= -DP3DFFT
  SOURCES_CPP+=fft3dmpi_with_p3dfft.cpp
endif

# LDFLAGS:=#-fpermissive

OBJECTS := $(SOURCES_CPP:.cpp=.o)
ifeq "$(CUDA)" "yes"
OBJECTS += $(SOURCES_CU:.cu=.o)
endif

empty :=
space := $(empty) $(empty)

SOURCES_CPP_BASE := $(wildcard ../base/*.cpp)
OBJECTS_CPP_BASE := $(subst ../base/,,$(SOURCES_CPP_BASE))
OBJECTS_CPP_BASE := $(OBJECTS_CPP_BASE:.cpp=.o)

OBJECTS_BUILD = build/$(subst $(space),$(space)build/,$(OBJECTS_CPP_BASE) $(OBJECTS))

red="\033[0;31m"
end="\033[0m"


.PHONY: help all clean _build_dir

all: build test_bench.out

show_targets:
	@echo -e $(red)$(SOURCES_CPP_BASE)$(end)
	@echo $(OBJECTS_CPP_BASE)
	@echo -e $(red)$(SOURCES_CPP)$(end)
	@echo $(OBJECTS)
	@echo -e $(red)include:$(end)
	@echo $(INC_DIR)
	@echo -e $(red)build:$(end)
	@echo $(OBJECTS_BUILD)
	@echo -e $(red)link:$(end)
	@echo $(LDFLAGS) $(OBJECTS_BUILD) $(LIBS) $(LIB_DIR)

build:
	mkdir -p build

clean:
	rm -rf build
	rm -f *.out

test_bench.out: $(OBJECTS) $(OBJECTS_CPP_BASE)
	# link to build the executable $@
	$(LD) $(LDFLAGS) $(OBJECTS_BUILD) $(LIBS) $(LIB_DIR) -o $@

%.o: %.cu
	# make $@ from $<
	nvcc $(FLAGS_CU) $(INC_DIR) $(INC_DIR_CU) -c $< -o build/$@

%.o: %.cpp
	# make $@ from $<
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o build/$@

%.o: %.cpp %.h
	# make $@ from $<
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o build/$@

%.o: ../base/%.cpp
	# make $@ from $<
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o build/$@

test:
	./test_bench.out --N0=64 --N1=32 --N2=16

testmpi:
	mpirun -np 4 ./test_bench.out --N0=64 --N1=32 --N2=16
	# mpirun -np 4 ./test_bench.out --N0=128 --N1=128 --N2=128
	# mpirun -np 4 ./test_bench.out --N0=256 --N1=256 --N2=256
	# mpirun -np 4 ./test_bench.out --N0=512 --N1=512 --N2=512
