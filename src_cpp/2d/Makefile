
CC:=mpic++
LIBS:=-lfftw3 -lfftw3_mpi
LDFLAGS:=-Wall

SOURCES_CPP := $(wildcard *.cpp)
OBJECTS := $(SOURCES_CPP:.cpp=.o)

empty :=
space := $(empty) $(empty)

SOURCES_CPP_BASE := $(wildcard ../base/*.cpp)
OBJECTS_CPP_BASE := $(subst ../base/,,$(SOURCES_CPP_BASE))
OBJECTS_CPP_BASE := $(OBJECTS_CPP_BASE:.cpp=.o)

OBJECTS_BUILD = build/$(subst $(space), $(space)build/,$(OBJECTS) $(OBJECTS_CPP_BASE))


.PHONY: help all clean _build_dir

all: test_bench.out

clean:
	rm -rf build
	rm -f *.out

_build_dir:
	mkdir -p build

test_bench.out: _build_dir $(OBJECTS) $(OBJECTS_CPP_BASE)
	# link to build the executable $@
	mpic++ $(OBJECTS_BUILD) $(LIBS) -o $@

%.o: %.cpp
	# make $@ from $<
	$(CC) $(LDFLAGS) -I. -I../base -c $< -o build/$@

%.o: %.cpp %.h
	# make $@ from $<
	$(CC) $(LDFLAGS) -I. -I../base -c $< -o build/$@

%.o: ../base/%.cpp
	# make $@ from $<
	$(CC) $(LDFLAGS) -I. -I../base -c $< -o build/$@
