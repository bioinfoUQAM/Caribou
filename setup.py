from setuptools import setup, find_packages
from metagenomics_ML import __version__

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='metagenomics_ML',
    version=_version,
    description='Alignment-free bacterial classification in metagenomic shotguns',
    author='Nicolas de Montigny',
    author_email='de_montigny.nicolas@courrier.uqam.ca',
    packages=find_packages(),
    include_packages_data=True,
    scripts=['metagenomics_ML/main.py',
            'metagenomics_ML/main_testing_hpc.py'],
    install_requires=INSTALL_REQUIRES
)
