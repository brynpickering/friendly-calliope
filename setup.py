#!/usr/bin/env python3

from pathlib import Path

from setuptools import setup, find_packages

requirements = Path("requirements.txt").read_text().strip().split("\n")

setup(
    name='friendly-calliope',
    version='0.3.0',
    description='Toolkit to prepare Calliope output data for dumping to friendly-data.',
    maintainer='Bryn Pickering',
    maintainer_email='bryn.pickering@usys.ethz.ch',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering'
    ]
)
