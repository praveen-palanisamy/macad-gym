#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="macad-gym",
    description='Learning environments for Multi-Agent Connected Autonomous'
    'Driving (MACAD)',
    version='0.0.1-0',
    url='https://github.com/praveen-palanisamy/macad-gym',
    author='Praveen Palanisamy',
    author_email='praveen.palanisamy@outlook.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.0',
    install_requires=[
        'gym', 'carla>=0.9.3', 'GPUtil', 'pygame', 'opencv-python', 'networkx'
    ],
    extras_require={'test': ['tox', 'pytest', 'pytest-xdist', 'tox']})
