.PHONY: clean cleanall develop list-sessions tests doc

develop: sync
	pdm run pip install -e plugins/fluidfft-fftw --no-build-isolation -v

develop_mpi_with_fftw:
	pdm run pip install -e plugins/fluidfft-mpi_with_fftw --no-build-isolation -v

develop_fftwmpi:
	pdm run pip install -e plugins/fluidfft-fftwmpi --no-build-isolation -v

develop_pfft:
	pdm run pip install -e plugins/fluidfft-pfft --no-build-isolation -v

develop_p3dfft:
	pdm run pip install -e plugins/fluidfft-p3dfft --no-build-isolation -v

sync:
	pdm sync --clean

lock:
	pdm lock

clean:
	rm -rf build

cleanso:
	find src -name "*.so" -delete

cleanpythran:
	find src -name __pythran__ -type d -exec rm -rf "{}" +

cleanall: clean cleanso cleanpythran

black:
	pdm run black

format:
	pdm run black

tests:
	pytest -s tests

tests_mpi:
	mpirun -np 2 pytest -s tests

tests_mpi4:
	mpirun -np 4 pytest -s tests

clang-format:
	find src_cpp/ -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i

list-sessions:
	@nox --version 2>/dev/null || pip install nox
	@nox -l

doc:
	nox -s doc

# Catch-all target: route all unknown targets to nox sessions
%:
	@nox --version 2>/dev/null || pip install nox
	@nox -s $@
