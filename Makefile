
.PHONY: clean cleanall cleanmako cleancython develop build_ext_inplace

develop:
	python setup.py develop

build_ext_inplace:
	python setup.py build_ext --inplace

clean:
	rm -rf build

cleanso:
	find fluidfft* -name "*.so" -delete

cleanpythran:
	find fluidfft* -name "*pythran*.so" -delete

cleancython:
	rm -f src_cy/*_cy.cpp

cleanmako:
	python -c "from src_cy.make_files_with_mako import clean_files as c; c()"

cleanall: clean cleanso cleanmako cleancython

tests:
	python -m unittest discover

tests_mpi:
	mpirun -np 2 python -m unittest discover

tests_mpi4:
	mpirun -np 4 python -m unittest discover

_tests_coverage:
	mkdir -p .coverage
	coverage run -p -m unittest discover
	mpirun -np 2 coverage run -p -m unittest discover

_report_coverage:
	coverage combine
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"

coverage: _tests_coverage _report_coverage
