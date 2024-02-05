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
from shutil import rmtree

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
        "-e",
        ".",
        "--no-deps",
        "--no-build-isolation",
        "-v",
        silent=False,
    )

    session.install("plugins/fluidfft-builder")
    session.install("-e", "plugins/fluidfft-fftw", "--no-build-isolation", "-v")
    if with_mpi:
        session.install(
            "-e", "plugins/fluidfft-mpi_with_fftw", "--no-build-isolation", "-v"
        )
        session.install("-e", "plugins/fluidfft-fftwmpi", "--no-build-isolation", "-v")

    if with_cov:
        path_coverage = Path.cwd() / ".coverage"
        rmtree(path_coverage, ignore_errors=True)
        path_coverage.mkdir(exist_ok=True)

    def run_command(command, **kwargs):
        if with_cov:
            command += " --cov --cov-config=pyproject.toml --no-cov-on-fail --cov-report=term-missing --cov-append"
        session.run(*command.split(), **kwargs)

    command = "pytest -v -s tests"

    run_command(command, *session.posargs)
    run_command(command, *session.posargs, env={"TRANSONIC_NO_REPLACE": "1"})
    run_command("pytest -v plugins/fluidfft-fftw")

    if with_mpi:

        def test_plugin(package_name):
            if with_cov:
                command = "mpirun -np 2 --oversubscribe coverage run -p -m pytest -v -s --exitfirst"
            else:
                command = "mpirun -np 2 --oversubscribe pytest -v -s "

            command += f" plugins/{package_name}"
            session.run(*command.split(), external=True)

        test_plugin("fluidfft-mpi_with_fftw")
        test_plugin("fluidfft-fftwmpi")

    if with_cov:
        if with_mpi:
            session.run("coverage", "combine")
        session.run("coverage", "report")
        session.run("coverage", "xml")
        session.run("coverage", "html")


@nox.session
def doc(session):
    session.run_always(
        "pdm", "sync", "--clean", "-G", "doc", "--no-self", external=True
    )
    session.run_always(
        "python", "-c", "from fluidfft_builder import create_fake_modules as c; c()"
    )
    session.install(".", "--no-deps", "-C", "setup-args=-Dtransonic-backend=python")
    session.chdir("doc")
    session.run("make", "cleanall", external=True)
    session.run("make", external=True)
