from setuptools import setup, find_packages

setup(
	name="seq_models",
	version="0.0.1",
	packages=find_packages(include=["seq_models", "seq_models.*"]),
)
