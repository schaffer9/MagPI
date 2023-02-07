from setuptools import setup, find_packages

setup(
    name="pinns",
    version="0.0.1",
    author="Sebastian Schaffer",
    packages=find_packages(
        exclude=["test"]
    )
)