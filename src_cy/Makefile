

.PHONY: help all clean clean_cython

all:
	python make_files_with_mako.py

clean_all: clean clean_cython clean_mako

clean_cython:
	# remove files produced by cython
	rm -f *_cy.cpp *_cy.c

clean_mako:
	# remove files produced by mako
	rm -f *_cy.pyx *_pxd.pxd

mako:
	python make_files_with_mako.py
