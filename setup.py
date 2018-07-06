# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_ = f.read()

desc = 'Machine learning models for feature-based Natural Language Inference'

setup(
    name='infernal',
    version='1.0.0',
    description=desc,
    long_description=readme,
    author='Erick Fonseca',
    author_email='erickrfonseca@gmail.com',
    url='https://github.com/erickrf/infernal',
    license=license_,
    packages=find_packages(exclude=('tests', 'docs'))
)