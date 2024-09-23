from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup( 
    name="model",
    version="0.1",
    packages=find_packages(),
    author="Jakub Iliński",
    author_email="kubailinski2@gmail.com",
    description=(
        "Comprehensive package for the YachtSpeedPred project, including "
        "TensorFlow models, supporting classes, and essential files for predicting "
        "yacht speed based on various parameters."
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=required,
    python_requires='>=3.10',
)