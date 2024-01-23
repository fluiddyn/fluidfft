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
def tests(session):
    """Execute unit-tests using pytest"""

    session.run_always(
        "pdm", "sync", "-G", "test", "--clean", "--no-self", external=True
    )
    session.install("-v", "-e", ".", "--force-reinstall", "--no-deps", silent=False)
    session.run("ls", "src/fluidfft/fft3d", silent=False, external=True)

    session.run(
        "python",
        "-m",
        "pytest",
        "-v",
        "-s",
        *session.posargs,
    )


@no_venv_session(name="tests-cov")
def tests_cov(session):
    """Execute unit-tests using pytest+pytest-cov"""
    session.notify(
        "tests",
        [
            "--cov",
            "--cov-config=setup.cfg",
            "--no-cov-on-fail",
            "--cov-report=term-missing",
            *session.posargs,
        ],
    )


@nox.session(name="coverage-html")
def coverage_html(session, nox=False):
    """Generate coverage report in HTML. Requires `tests-cov` session."""
    report = Path.cwd() / ".coverage" / "html" / "index.html"
    session.install("coverage[toml]", "cython")
    session.run("coverage", "html")


@nox.session(name="tests-full")
def tests_full(session):
    """Execute all unit-tests using pytest"""

    session.run_always(
        "pdm", "sync", "-G", "test", "-G", "mpi", "--clean", "--no-self", external=True
    )

    session.install("-v", "-e", ".", "--force-reinstall", "--no-deps", silent=False)
    session.run("ls", "src/fluidfft/fft3d", silent=False, external=True)

    cov_path = Path.cwd() / ".coverage"
    cov_path.mkdir(exist_ok=True)

    def run_command(command, **kwargs):
        words = command.split()
        session.run(
            *words,
            **kwargs,
        )

    run_command("coverage run -p -m pytest -s src")
    run_command("coverage run -p -m pytest -s src", env={"TRANSONIC_NO_REPLACE": "1"})
    # Using TRANSONIC_NO_REPLACE with mpirun in docker can block the tests
    run_command(
        "mpirun -np 2 --oversubscribe coverage run -p -m unittest discover src",
        external=True,
    )
