from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup( 
    name="model",
    version="0.1",
    packages=find_packages(),
    author="Jakub Iliński",
    description="Comprehensive package for the YachtSpeedPred project, including TensorFlow models, supporting classes and essential files for predicting yacht speed based on various parameters.",
    install_requires=required,
)