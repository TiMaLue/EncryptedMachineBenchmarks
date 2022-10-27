import pathlib
import setuptools

HERE = pathlib.Path(__file__).parent


setuptools.setup(
    name="commonsnakes",
    version="1.0",
    author="Amin Faez",
    author_email="amin.faez.inbox@gmail.com",
    classifiers=[
        "Utilities",
        "Programming Language :: Python"
    ],
    install_requires=[
        "pyyaml"
    ],
    packages=[package for package in setuptools.find_packages() if "test" not in package],
    python_requires=">=3.6"
)
