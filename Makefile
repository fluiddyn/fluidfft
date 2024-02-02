.PHONY: clean cleanall develop build_ext_inplace list-sessions tests

develop: sync
	pdm run pip install -e . --no-deps --no-build-isolation -v
	pdm run pip install -e plugins/fluidfft-fftw --no-build-isolation -v

develop_mpi_with_fftw:
	pdm run pip install -e plugins/fluidfft-mpi_with_fftw --no-build-isolation -v

sync:
	pdm sync --clean --no-self
	pdm run pip install -e plugins/fluidfft-builder

clean:
	rm -rf build

cleanso:
	find src -name "*.so" -delete

cleanpythran:
	find src -name __pythran__ -type d -exec rm -rf "{}" +

cleanall: clean cleanso cleanpythran

black:
	pdm run black

tests:
	pytest -s tests

tests_mpi:
	mpirun -np 2 pytest -s tests

tests_mpi4:
	mpirun -np 4 pytest -s tests

_tests_coverage:
	mkdir -p .coverage
	coverage run -p -m pytest -s tests
	TRANSONIC_NO_REPLACE=1 coverage run -p -m pytest -s tests
	# Using TRANSONIC_NO_REPLACE with mpirun in docker can block the tests
	mpirun -np 2 --oversubscribe coverage run -p -m unittest discover tests

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

lock:
	pdm lock -G :all

# Catch-all target: route all unknown targets to nox sessions
%:
	@nox --version 2>/dev/null || pip install nox
	@nox -s $@
