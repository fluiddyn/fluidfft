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

    command = "pdm sync --clean --no-self -G test -G build -G pyfftw"
    if with_mpi:
        command += " -G mpi"
    session.run_always(*command.split(), external=True)

    session.install(
        ".",
        "--no-deps",
        "-v",
        silent=False,
    )
    session.run("ls", "src/fluidfft/fft3d", silent=False, external=True)

    session.install("plugins/fluidfft-builder")
    session.install("-e", "plugins/fluidfft-fftw", "--no-build-isolation", "-v")
    if with_mpi:
        session.install(
            "-e", "plugins/fluidfft-mpi_with_fftw", "--no-build-isolation", "-v"
        )
        session.install("-e", "plugins/fluidfft-fftwmpi", "--no-build-isolation", "-v")

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

    run_command("pytest -v plugins/fluidfft-fftw")

    if with_mpi:
        if with_cov:
            command = "mpirun -np 2 --oversubscribe coverage run -p -m pytest -v -s --exitfirst tests"

        else:
            command = "mpirun -np 2 --oversubscribe pytest -v -s tests"

        run_command(command, external=True)

        run_command(
            "mpirun -np 2 --oversubscribe pytest -v plugins/fluidfft-mpi_with_fftw",
            external=True,
        )
        run_command(
            "mpirun -np 2 --oversubscribe pytest -v plugins/fluidfft-fftwmpi",
            external=True,
        )

    if with_cov:
        if with_mpi:
            run_command("coverage combine")
        run_command("coverage report")


@nox.session
def doc(session):
    session.run_always("pdm", "sync", "--clean", "-G", "doc", "--no-self", external=True)
    session.run_always("python", "-c", "from fluidfft_builder import create_fake_modules as c; c()")
    session.install(
        ".", "--no-deps", "-C", "setup-args=-Dtransonic-backend=python"
    )
    session.chdir("doc")
    session.run("make", "cleanall", external=True)
    session.run("make", external=True)
