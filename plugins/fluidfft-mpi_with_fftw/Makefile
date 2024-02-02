
develop:
	pip install -e . -vv --no-build-isolation --no-deps

clean:
	rm -rf build

test:
	mpirun -np 2 pytest --exitfirst -v
