
.PHONY: clean clean_all clean_mako clean_cython develop build_ext_inplace mako

develop: mako
	python setup.py develop

build_ext_inplace: mako
	python setup.py build_ext --inplace

clean:
	rm -rf build

clean_so:
	find fluidfft* -name "*.so" -delete

clean_cython:
	rm -f src_cy/*_cy.cpp

clean_mako:
	rm -f src_cy/*_cy.pyx
	rm -f src_cy/*_pxd.pxd

clean_all: clean clean_so clean_mako clean_cython

mako:
	cd src_cy && python make_files_with_mako.py
