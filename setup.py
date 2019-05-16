#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="macad-gym",
      description='Learning environments for Multi-Agent Connected Autonomous'
      'Driving (MACAD)',
      url='https://github.com/praveen-palanisamy/macad-gym',
      author='Praveen Palanisamy',
      author_email='praveen.palanisamy@outlook.com',
      packages=find_packages(),
      package_dir={'': 'src'},
      version='0.0.1-dev'
      )
