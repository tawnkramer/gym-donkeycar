#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import find_packages, setup

with open(os.path.join("gym_donkeycar", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

description = "OpenAI Gym Environments for Donkey Car"


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["gym", "numpy", "pillow"]


setup(
    name="gym_donkeycar",
    author="Tawn Kramer",
    author_email="tawnkramer@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description=description,
    install_requires=requirements,
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-mock",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
    },
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="donkeycar, environment, agent, rl, openaigym, openai-gym, gym",
    packages=find_packages(),
    url="https://github.com/tawnkramer/gym-donkeycar",
    version=__version__,
    zip_safe=False,
)
