from setuptools import setup, find_packages

setup(
    name="magpi",
    version="0.0.1",
    author="Sebastian Schaffer",
    packages=find_packages(
        exclude=["test"]
    ),
    python_requires='>=3.11'
)
