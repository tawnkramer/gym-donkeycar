#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from gym_donkeycar import __version__

"""The setup script."""

description = 'OpenAI Gym Environments for Donkey Car'


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'gym',
    'numpy',
    'pillow'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Leigh Johnson",
    author_email='leigh@data-literate.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description=description,
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='gym_donkeycar',
    name='gym_donkeycar',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/leigh-johnson/gym_donkeycar',
    version=__version__,
    zip_safe=False,
)
