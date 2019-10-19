from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    dependencies = [line.strip().split(" ")[0] for line in f.readlines()]

print(dependencies)

setup(
    name="rstools",
    version="1.0",
    description="A useful module",
    license="MIT",
    long_description=long_description,
    author="Nik Vaessen",
    author_email="vaessen@kth.se",
    packages=["rstools"],  # same as name
    install_requires=dependencies,  # external packages as dependencies
    scripts=["scripts/rs_record_now_here.py"],
)
