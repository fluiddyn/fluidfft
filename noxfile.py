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
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["tests"]

no_venv_session = partial(nox.session, venv_backend="none")


@nox.session(reuse_venv=True)
def validate_code(session):
    """Validate the code with black and pylint"""
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
    """Build the documentation"""
    session.run_always(
        "pdm", "sync", "--clean", "-G", "doc", "--no-self", external=True
    )
    session.run_always(
        "python",
        "-c",
        "from fluidfft_builder import create_fake_modules as c; c()",
    )
    session.install(
        ".", "--no-deps", "-C", "setup-args=-Dtransonic-backend=python"
    )
    session.chdir("doc")
    session.run("make", "cleanall", external=True)
    session.run("make", external=True)


def _get_version_from_pyproject(path=Path.cwd()):
    if isinstance(path, str):
        path = Path(path)

    if not path.name == "pyproject.toml":
        path /= "pyproject.toml"

    if not path.exists():
        raise IOError(f"{path} does not exist.")

    in_project = False
    version = None
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("[project]"):
                in_project = True
            if line.startswith("version =") and in_project:
                version = line.split("=")[1].strip()
                version = version[1:-1]
                break

    assert version is not None
    return version


@nox.session(name="add-tag-for-release", venv_backend="none")
def add_tag_for_release(session):
    """Add a version tag in the repo"""
    session.run("hg", "pull", external=True)

    result = session.run(
        *"hg log -r default -G".split(), external=True, silent=True
    )
    if result[0] != "@":
        session.run("hg", "update", "default", external=True)

    version = _get_version_from_pyproject()
    print(f"{version = }")

    result = session.run("hg", "tags", "-T", "{tag},", external=True, silent=True)
    last_tag = result.split(",", 2)[1]
    print(f"{last_tag = }")

    if last_tag == version:
        session.error("last_tag == version")

    answer = input(
        f'Do you really want to add and push the new tag "{version}"? (yes/[no]) '
    )

    if answer != "yes":
        print("Maybe next time then. Bye!")
        return

    print("Let's go!")
    session.run("hg", "tag", version, external=True)
    session.run("hg", "push", external=True)


@nox.session(name="release-plugin", reuse_venv=True)
def release_plugin(session):
    """Release a plugin on PyPI"""

    for project in ("build", "twine", "lastversion"):
        session.install(project)

    try:
        short_name = session.posargs[0]
    except IndexError:
        session.error(
            "No short name given. Use as `nox -R -s release-plugin -- fftw`"
        )
    print(short_name)

    path = Path.cwd() / f"plugins/fluidfft-{short_name}"

    if not path.exists():
        session.error(f"{path} does not exist")

    version = _get_version_from_pyproject(path)
    print(f"{version = }")

    ret = session.run(
        "lastversion",
        f"fluidfft-{short_name}",
        "--at",
        "pip",
        success_codes=[0, 1],
        silent=True,
    )
    if ret.startswith("CRITICAL: No release was found"):
        print(ret[len("CRITICAL: ") :])
    else:
        version_on_pypi = ret.strip()
        if version_on_pypi == version:
            session.error(f"Local version {version} is already released")

    session.chdir(path)

    path_dist = path / "dist"
    rmtree(path_dist, ignore_errors=True)

    command = "python -m build"
    if short_name in ["fftw", "mpi_with_fftw", "fftwmpi", "pfft", "p3dfft"]:
        command += " --sdist"

    session.run(*command.split())
    session.run("twine", "check", "dist/*")

    answer = input(
        f"Do you really want to release fluidfft-{short_name} {version}? (yes/[no]) "
    )

    if answer != "yes":
        print("Maybe next time then. Bye!")
        return

    session.run("twine", "upload", "dist/*")


@nox.session(name="create-fake-modules", reuse_venv=True)
def create_fake_modules(session):
    """Create fake modules for doc"""

    session.install("-e", "./plugins/fluidfft-builder")
    session.install("black")
    session.run(
        "python",
        "-c",
        "from fluidfft_builder import create_fake_modules as c; c()",
    )
    session.run("black", "src/fluidfft")


@nox.session(python=False)
def detect_pythran_extensions(session):
    """Detect and print Pythran extension modules"""
    session.chdir("src")
    begin = "- "
    # begin = "import "
    paths_pythran_files = sorted(Path("fluidfft").rglob("*/__pythran__/*.py"))
    print(
        begin
        + f"\n{begin}".join(
            [str(p)[:-3].replace("/", ".") for p in paths_pythran_files]
        )
    )
