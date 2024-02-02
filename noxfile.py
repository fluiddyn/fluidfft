"""Task runner for the developer

Usage
-----

   nox -l

   nox -s <session>

   nox -k <keyword>
or:

   make <session>

execute ``make list-sessions```` or ``nox -l`` for a list of sessions.

"""

import os
import shlex
from pathlib import Path
from functools import partial

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
nox.options.reuse_existing_virtualenvs = 1
nox.options.sessions = ["tests"]

no_venv_session = partial(nox.session, venv_backend="none")


@nox.session
def validate_code(session):
    session.run_always(
        "pdm", "sync", "--clean", "-G", "lint", "--no-self", external=True
    )
    session.run("pdm", "validate_code", external=True)


@nox.parametrize("with_mpi", [True, False])
@nox.parametrize("with_cov", [True, False])
@nox.session
def tests(session, with_mpi, with_cov):
    """Execute unit-tests using pytest"""

    command = "pdm sync --clean --no-self -G test"
    if with_mpi:
        command += " -G mpi"
    session.run_always(*command.split(), external=True)

    session.install(
        ".", "--no-deps", "-v",
        "--config-settings=setup-args=-Dtransonic-backend=python",
        silent=False,
    )
    session.run("ls", "src/fluidfft/fft3d", silent=False, external=True)

    session.install("-e", "plugins/fluidfft-builder")
    session.install("-e", "plugins/fluidfft-fftw")
    if with_mpi:
        session.install("-e", "plugins/fluidfft-mpi_with_fftw")

    def run_command(command, **kwargs):
        session.run(*command.split(), **kwargs)

    if with_cov:
        cov_path = Path.cwd() / ".coverage"
        cov_path.mkdir(exist_ok=True)

    command = "pytest -v -s tests"
    if with_cov:
        command += (
            " --cov --cov-config=setup.cfg --no-cov-on-fail --cov-report=term-missing"
        )

    run_command(command, *session.posargs)
    run_command(command, *session.posargs, env={"TRANSONIC_NO_REPLACE": "1"})

    if with_mpi:
        if with_cov:
            command = "mpirun -np 2 --oversubscribe coverage run -p -m pytest -v -s --exitfirst tests"

        else:
            command = "mpirun -np 2 --oversubscribe pytest -v -s tests"

        # Using TRANSONIC_NO_REPLACE with mpirun in docker can block the tests
        run_command(command, external=True)

    if with_cov:
        if with_mpi:
            run_command("coverage combine")
        run_command("coverage report")


@nox.session
def doc(session):
    session.run_always("pdm", "sync", "-G", "doc", "--no-self", external=True)
    session.install(
        ".", "--no-deps", "--config-settings=setup-args=-Dtransonic-backend=python"
    )
    session.install("-e", "plugins/fluidfft-pyfftw")

    session.chdir("doc")
    session.run("make", "cleanall", external=True)
    session.run("make", external=True)
