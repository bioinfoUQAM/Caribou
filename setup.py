from setuptools import setup, find_packages
from Caribou import __version__

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='Caribou',
    version=_version,
    description='Alignment-free bacterial classification in metagenomic shotguns',
    author='Nicolas de Montigny',
    author_email='nicolas.de.montingy0323@gmail.com',
    packages=find_packages(),
    scripts=[],
    install_requires=INSTALL_REQUIRES
)
