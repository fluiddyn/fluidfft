
.PHONY: clean cleanall cleanmako cleancython develop build_ext_inplace

develop:
	pip install -v -e .[dev] | grep -v link

clean:
	rm -rf build

cleanso:
	find fluidfft -name "*.so" -delete

cleanpythran:
	find fluidfft -name __pythran__ -type d -exec rm -rf "{}" +

cleancython:
	find fluidfft -name "*_cy.cpp" -delete

cleanmako:
	python -c "from src_cy.make_files_with_mako import clean_files as c; c()"

cleanall: clean cleanso cleanmako cleancython cleanpythran

black:
	black -l 82 fluidfft

tests:
	pytest -s

tests_mpi:
	mpirun -np 2 pytest -s

tests_mpi4:
	mpirun -np 4 pytest -s

_tests_coverage:
	mkdir -p .coverage
	coverage run -p -m unittest discover
	TRANSONIC_NO_REPLACE=1 coverage run -p -m unittest discover
	TRANSONIC_NO_REPLACE=1 mpirun -np 2 coverage run -p -m unittest discover

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