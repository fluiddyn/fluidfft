# How to build and run:
# ---------------------
# make clean all test testmpi
FLUIDFFT_SRC_CPP:=$(realpath ../../src_cpp)
MAIN:=compute_energy

SINGLE_PREC:=no
OMP:=no
CUDA:=no
PFFT:=yes
P3DFFT:=yes

LD:=mpic++
CC:=mpic++
FLAGS:= -g -O3 -Wall
LDFLAGS:=
LIBS:= -ldl -lhwloc -lm -lgfortran

# Ensure FFTW3, P3DFFT, PFFT library and include directories are specified
# either below or in LD_LIBRARY_PATHS and CPATH
LIB_DIR:=$(LDEV_PATH)
INC_DIR:=-I. -I$(FLUIDFFT_SRC_CPP)/base -I$(FLUIDFFT_SRC_CPP)/3d $(IDEV_PATH)

SOURCES_CPP := $(wildcard $(FLUIDFFT_SRC_CPP)/3d/base*.cpp)
SOURCES_CPP += $(FLUIDFFT_SRC_CPP)/3d/fft3d_with_fftw3d.cpp $(FLUIDFFT_SRC_CPP)/3d/fft3dmpi_with_fftw1d.cpp $(FLUIDFFT_SRC_CPP)/3d/fft3dmpi_with_fftwmpi3d.cpp $(MAIN).cpp

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
  SOURCES_CPP+=$(FLUIDFFT_SRC_CPP)/3d/fft3dmpi_with_pfft.cpp
endif

ifeq "$(P3DFFT)" "yes"
  FLAGS+= -DP3DFFT
  SOURCES_CPP+=$(FLUIDFFT_SRC_CPP)/3d/fft3dmpi_with_p3dfft.cpp
endif

# LDFLAGS:=#-fpermissive

OBJECTS := $(SOURCES_CPP:.cpp=.o)
ifeq "$(CUDA)" "yes"
OBJECTS += $(SOURCES_CU:.cu=.o)
endif
OBJECTS := $(subst $(FLUIDFFT_SRC_CPP)/3d/,,$(OBJECTS))
# OBJECTS := $(subst $(FLUIDFFT_SRC_CPP)/base/,,$(OBJECTS))

empty :=
space := $(empty) $(empty)

SOURCES_CPP_BASE := $(wildcard $(FLUIDFFT_SRC_CPP)/base/*.cpp)
OBJECTS_CPP_BASE := $(subst $(FLUIDFFT_SRC_CPP)/base/,,$(SOURCES_CPP_BASE))
OBJECTS_CPP_BASE := $(OBJECTS_CPP_BASE:.cpp=.o)
OBJECTS_BUILD += build/$(subst $(space),$(space)build/,$(OBJECTS_CPP_BASE) $(OBJECTS))

red="\033[0;31m"
end="\033[0m"


.PHONY: help show_targets all clean build test testmpi

all: build $(MAIN).out

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

$(MAIN).out: $(OBJECTS) $(OBJECTS_CPP_BASE)
	@echo -e $(red)link to build the executable $@$(end)
	$(LD) $(LDFLAGS) $(OBJECTS_BUILD) $(LIBS) $(LIB_DIR) -o $@

%.o: %.cu
	@echo -e $(red)make $@ from $<$(end)
	nvcc $(FLAGS_CU) $(INC_DIR) $(INC_DIR_CU) -c $< -o build/$@

%.o: %.cpp
	@echo -e $(red)make $@ from $<$(end)
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o build/$@

%.o: $(FLUIDFFT_SRC_CPP)/3d/%.cpp $(FLUIDFFT_SRC_CPP)/3d/%.h
	@echo -e $(red)make $@ from $<$(end)
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o build/$@

%.o: $(FLUIDFFT_SRC_CPP)/base/%.cpp
	@echo -e $(red)make $@ from $<$(end)
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o build/$@

test:
	./$(MAIN).out

testmpi:
	mpirun -np 2 ./$(MAIN).out
