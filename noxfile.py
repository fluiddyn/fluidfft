"""Task runner for the developer

Usage
-----

   nox -l

   nox -s <session>

   nox -k <keyword>
or:

   make <session>

execute `make list-sessions` or `nox -l` for a list of sessions.

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


@nox.session(reuse_venv=True)
def validate_code(session):
    session.run_always(
        "pdm", "sync", "--clean", "-G", "lint", "--no-self", external=True
    )
    session.run("pdm", "validate_code", external=True)


@nox.parametrize("with_mpi", [True, False])
@nox.parametrize("with_cov", [True, False])
@nox.session(reuse_venv=True)
def tests(session, with_mpi, with_cov):
    """Execute unit-tests using pytest"""

    with_pfft = "--with-pfft" in session.posargs
    with_p3dfft = "--with-p3dfft" in session.posargs

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

    plugin_name_seq = ["fftw"]
    plugin_names = plugin_name_seq.copy()

    if with_mpi:
        plugin_names_par = ["mpi_with_fftw", "fftwmpi"]
        if with_pfft:
            plugin_names_par.append("pfft")
        if with_p3dfft:
            plugin_names_par.append("p3dfft")
        plugin_names.extend(plugin_names_par)

    for name in plugin_names:
        session.install("-e", f"plugins/fluidfft-{name}", "--no-build-isolation")

    if with_cov:
        path_coverage = Path.cwd() / ".coverage"
        rmtree(path_coverage, ignore_errors=True)
        path_coverage.mkdir(exist_ok=True)

    def run_command(command, **kwargs):
        if with_cov:
            command += " --cov --cov-config=pyproject.toml --no-cov-on-fail --cov-report=term-missing --cov-append"
        session.run(*command.split(), **kwargs)

    command = "pytest -v -s tests"

    run_command(command)
    run_command(command, env={"TRANSONIC_NO_REPLACE": "1"})

    for name in plugin_name_seq:
        run_command(f"pytest -v plugins/fluidfft-{name}")

    if with_mpi:

        def test_plugin(package_name):
            if with_cov:
                command = "mpirun -np 2 --oversubscribe coverage run -p -m pytest -v -s --exitfirst"
            else:
                command = "mpirun -np 2 --oversubscribe pytest -v -s "

            command += f" plugins/{package_name}"
            session.run(*command.split(), external=True)

        for name in plugin_names_par:
            test_plugin(f"fluidfft-{name}")

    if with_cov:
        if with_mpi:
            session.run("coverage", "combine")
        session.run("coverage", "report")
        session.run("coverage", "xml")
        session.run("coverage", "html")


@nox.session(reuse_venv=True)
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
