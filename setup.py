#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="Higher-Order Interactions",
    author="Tim Liu",
    author_email="zl5216@ic.ac.uk",
    version="0.0.1",
    description="Code to compute the higher-order interactions in observational data",
    install_requires=["pandas", "numpy", "networkx", "matplotlib", "tqdm", "scikit-learn"],
    packages=find_packages(),
)
