"""The compiled module imports and reports the version of the installed package."""
from importlib.metadata import version

import nes_cpp


def test_version_matches_package_metadata():
    assert nes_cpp.__version__ == version("nes-cpp")
