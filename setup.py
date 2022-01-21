from setuptools import setup, find_namespace_packages

__author__ = "Nicolas de Montigny"

__all__ = []

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name = 'Caribou',
    version = "1.0.0",
    description = 'Alignment-free bacterial classification in metagenomic shotguns',
    author = 'Nicolas de Montigny',
    author_email = 'de_montigny.nicolas@courrier.uqam.ca',
    python_requires=">=3.8",
    packages = find_namespace_packages(),
    namespace_packages=['Caribou'],
    license = 'MIT license',
    package_data={'Caribou.data.KMC': ['bin/kmc',
                  'bin/kmc_dump',
                  'bin/kmc_tools',
                  'bin/libkmc_core.a',
                  'include/kmc_runner.h'], 'Caribou.data':['faSplit']},
    include_package_data = True,
    scripts = ['Caribou/Caribou.py',
                'Caribou/main_testing_hpc.py',
                'Caribou/K-mers_extract.py'
                'Caribou/data/build_data.py',
                'Caribou/models/bacteria_extraction.py',
                'Caribou/models/classification.py',
                'Caribou/outputs/outputs.py',
                'Caribou/supplement/R_interface.R',
                'Caribou/supplement/simulation.py'],
    install_requires = INSTALL_REQUIRES
)
