"""Setups the project."""
from path import Path
from setuptools import setup, find_packages
import os


CWD = Path(os.path.abspath(os.path.dirname(__file__)))


def get_version():
    """Gets the pettingzoo version."""
    path = CWD / "pettingzoo" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    name="carla-gym",
    version=get_version(),
    url='https://github.com/johnMinelli/macad-gym',
    packages=find_packages('carla_gym'),
    package_dir={'': 'carla_gym'},
    python_requires='>=3.0',
    install_requires=[
        'gym', 'carla>=0.9.3', 'GPUtil', 'pygame', 'opencv-python', 'networkx'
    ],
    extras_require={'test': ['tox', 'pytest', 'pytest-xdist', 'tox']},
)
