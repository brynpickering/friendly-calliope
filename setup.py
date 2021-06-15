#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='friendly-calliope',
    version='0.1.0',
    description='Toolkit to prepare Calliope output data for dumping to friendly-data.',
    maintainer='Bryn Pickering',
    maintainer_email='bryn.pickering@usys.ethz.ch',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    install_requires=[
        "numpy",
        "calliope>=0.6.6",
        "pandas",
        "xarray>=0.18",
        "pyyaml"
    ],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering'
    ]
)
