from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Radial Basis Function Interpolation in PyTorch'

# Setting up
setup(
    name="torchrbf",
    version=VERSION,
    author="Arman Maesumi",
    author_email="arman.maesumi@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'pytorch', 'rbf', 'interpolation', 'radial basis function'],
    classifiers=[]
)