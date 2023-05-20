"""Setups the project."""
from setuptools import setup, find_packages
import os


CWD = os.path.abspath(os.path.dirname(__file__))


def get_version():
    """Gets the lib version."""
    path = os.path.join(CWD, "carla_gym", "__init__.py")
    with open(path) as f:
        content = f.read()
        for line in content.splitlines():
            if line.startswith("__version__"):
                return line.strip().split()[-1].strip().strip('"')
        raise RuntimeError("bad version data in __init__.py")


# Prepare long description using existing docs
long_description = ""
with open("README.md") as f:
    long_description = "\n".join([long_description, f.read()])
# Replace relative path to images with GitHub URI
github_uri_prefix = "https://raw.githubusercontent.com/johnMinelli/carla-gym/master/"
rel_img_path = "docs/images/"
long_description = long_description.replace("(" + rel_img_path, "(" + github_uri_prefix + rel_img_path)


setup(
    name="carla-gym",
    version=get_version(),
    license="MIT",
    description="Learning environments for Multi-Agent Connected Autonomous Driving (MACAD) with OpenAI Gym compatible interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnMinelli/carla-gym",
    packages=[package for package in find_packages() if package.startswith("carla_gym")],
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    package_data={"carla_gym": ["*.xml"]},
    install_requires=["gym", "path", "carla>=0.9.3", "GPUtil", "pygame", "opencv-python", "networkx"],
    extras_require={"test": ["tox", "pytest", "pytest-xdist", "tox"]},
)
