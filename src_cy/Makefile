

.PHONY: help all clean clean_cython

all:
	python make_files_with_mako.py

cleanall: cleancython cleanmako

cleancython:
	# remove files produced by cython
	rm -f *_cy.cpp *_cy.c

cleanmako:
	# remove files produced by mako
	rm -f *_cy.pyx *_pxd.pxd

mako:
	python make_files_with_mako.py
