
.PHONY: clean clean_all clean_mako clean_cython develop build_ext_inplace mako

develop:
	python setup.py develop

build_ext_inplace: mako
	python setup.py build_ext --inplace

clean:
	rm -rf build

cleanso:
	find fluidfft* -name "*.so" -delete

cleancython:
	rm -f src_cy/*_cy.cpp

cleanmako:
	rm -f src_cy/*_cy.pyx
	rm -f src_cy/*_pxd.pxd

cleanall: clean cleanso cleanmako cleancython

# mako:
# 	cd src_cy && python make_files_with_mako.py

tests:
	python -m unittest discover

tests_mpi:
	mpirun -np 2 python -m unittest discover

tests_coverage:
	mkdir -p .coverage
	coverage run -p -m unittest discover
	mpirun -np 2 coverage run -p -m unittest discover

report_coverage:
	coverage combine
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"

coverage: tests_coverage report_coverage
