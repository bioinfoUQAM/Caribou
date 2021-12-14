from setuptools import setup, find_namespace_packages
from Caribou import __version__

__author__ = "Nicolas de Montigny"

__all__ = []

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name = 'Caribou',
    version = _version,
    description = 'Alignment-free bacterial classification in metagenomic shotguns',
    author = 'Nicolas de Montigny',
    author_email = 'de_montigny.nicolas@courrier.uqam.ca',
    python_requires=">=3.8",
    packages = find_namespace_packages(),
    #package_data={'': ['*.pl', 'Caribou/Caribou/outputs/KronaTools/scripts/ImportText.pl']},
    namespace_packages=['Caribou'],
    license = 'LICENSE',
    include_package_data = True,
    scripts = ['Caribou/Caribou.py',
                'Caribou/main_testing_hpc.py'],
    install_requires = INSTALL_REQUIRES
)
