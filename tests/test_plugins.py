from importlib_metadata import entry_points

from fluidfft import get_plugins


def test_toto():

    plugins = get_plugins()
    assert plugins
