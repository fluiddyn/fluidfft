# How to build and run:
# ---------------------
# make clean all test testmpi
FLUIDFFT_SRC_CPP:=$(realpath ../../../src_cpp)
MAIN:=main

SINGLE_PREC:=no
OMP:=no
CUDA:=no

LD:=mpic++
CC:=mpic++
FLAGS:= -g -O3 -Wall
LDFLAGS:=
LIBS:= -ldl -lhwloc -lm -lgfortran

# Ensure FFTW3 library and include directories are specified
# either below or in LD_LIBRARY_PATHS and CPATH
LIB_DIR:=$(LDEV_PATH)
INC_DIR:=-I. -I$(FLUIDFFT_SRC_CPP)/base -I$(FLUIDFFT_SRC_CPP)/2d $(IDEV_PATH)

SOURCES_CPP := $(wildcard $(FLUIDFFT_SRC_CPP)/2d/base*.cpp)
SOURCES_CPP += $(FLUIDFFT_SRC_CPP)/2d/fft2d_with_fftw2d.cpp $(FLUIDFFT_SRC_CPP)/2d/fft2dmpi_with_fftw1d.cpp $(FLUIDFFT_SRC_CPP)/2d/fft2dmpi_with_fftwmpi2d.cpp $(MAIN).cpp

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
else
  LIBS+= -lfftw3 -lfftw3_mpi -lfftw3_omp
endif

# LDFLAGS:=#-fpermissive

OBJECTS := $(SOURCES_CPP:.cpp=.o)
ifeq "$(CUDA)" "yes"
OBJECTS += $(SOURCES_CU:.cu=.o)
endif
OBJECTS := $(subst $(FLUIDFFT_SRC_CPP)/2d/,,$(OBJECTS))

empty :=
space := $(empty) $(empty)

SOURCES_CPP_BASE := $(wildcard $(FLUIDFFT_SRC_CPP)/base/*.cpp)
OBJECTS_CPP_BASE := $(subst $(FLUIDFFT_SRC_CPP)/base/,,$(SOURCES_CPP_BASE))
OBJECTS_CPP_BASE := $(OBJECTS_CPP_BASE:.cpp=.o)

red="\033[0;31m"
end="\033[0m"


.PHONY: help show_targets all clean bug test testmpi

bug: $(MAIN).out
	mpirun -np 16 ./$(MAIN).out

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

clean:
	rm -f *.out *.o

$(MAIN).out: $(OBJECTS) $(OBJECTS_CPP_BASE)
	@echo -e $(red)link to build the executable $@$(end)
	$(LD) $(LDFLAGS) $(OBJECTS)  $(OBJECTS_CPP_BASE) $(LIBS) $(LIB_DIR) -o $@

%.o: %.cpp
	@echo -e $(red)make $@ from $<$(end)
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o $@

%.o: $(FLUIDFFT_SRC_CPP)/2d/%.cpp $(FLUIDFFT_SRC_CPP)/2d/%.h
	@echo -e $(red)make $@ from $<$(end)
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o $@

%.o: $(FLUIDFFT_SRC_CPP)/base/%.cpp
	@echo -e $(red)make $@ from $<$(end)
	$(CC) $(FLAGS) $(INC_DIR) -c $< -o $@

test: $(MAIN).out
	./$(MAIN).out

testmpi: $(MAIN).out
	mpirun -np 16 ./$(MAIN).out
