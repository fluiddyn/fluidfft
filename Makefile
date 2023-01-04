.PHONY: clean cleanall cleanmako cleancython develop build_ext_inplace list-sessions requirements

develop:
	pip install -v -e .[dev] | grep -v link

clean:
	rm -rf build

cleanso:
	find src -name "*.so" -delete

cleanpythran:
	find src -name __pythran__ -type d -exec rm -rf "{}" +

cleancython:
	find src -name "*_cy.cpp" -delete

cleanmako:
	python -c "from src_cy.make_files_with_mako import clean_files as c; c()"

cleanall: clean cleanso cleanmako cleancython cleanpythran

black:
	black -l 82 src *.py

tests:
	pytest -s src

tests_mpi:
	mpirun -np 2 pytest -s src

tests_mpi4:
	mpirun -np 4 pytest -s src

_tests_coverage:
	mkdir -p .coverage
	coverage run -p -m pytest -s src
	TRANSONIC_NO_REPLACE=1 coverage run -p -m pytest -s src
	# Using TRANSONIC_NO_REPLACE with mpirun in docker can block the tests
	mpirun -np 2 --oversubscribe coverage run -p -m unittest discover src

_report_coverage:
	coverage combine
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"

coverage: _tests_coverage _report_coverage

clang-format:
	find src_cpp/ -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i

list-sessions:
	@nox --version 2>/dev/null || pip install nox
	@nox -l

requirements: 'pip-compile(main)' 'pip-compile(doc)' 'pip-compile(test)' 'pip-compile(dev)'

# Catch-all target: route all unknown targets to nox sessions
%:
	@nox --version 2>/dev/null || pip install nox
	@nox -s $@