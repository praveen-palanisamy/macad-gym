#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# Prepare long description using existing docs
long_description = ""
this_dir = os.path.abspath(os.path.dirname(__file__))
doc_files = ["README.md"]
for doc in doc_files:
    with open(os.path.join(this_dir, doc), 'r') as f:
        long_description = "\n".join([long_description, f.read()])
# Replace relative path to images with Github URI
github_uri_prefix = "https://raw.githubusercontent.com/praveen-palanisamy" \
                    "/macad-gym/master/"
rel_img_path = "docs/images/"
long_description = long_description.replace(
    "(" + rel_img_path, "(" + github_uri_prefix + rel_img_path)

setup(
    name="macad-gym",
    version='0.1.0',
    description='Learning environments for Multi-Agent Connected Autonomous'
    'Driving (MACAD) with OpenAI Gym compatible interfaces',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
