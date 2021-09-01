import os
import sys
import setuptools
from setuptools.command.install import install

# The version of this package
VERSION = "0.1.15"


class VerifyVersionCommand(install):
    """
    Custom command to verify that the git tag matches the package version.
    Source: https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/
    """

    description = "verify that the git tag matches the package version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != VERSION:
            info = (
                f"Git tag: {tag} does not match the version of this package: {VERSION}"
            )
            sys.exit(info)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bavard-ml-common",
    version=VERSION,
    author="Bavard AI, Inc.",
    author_email="dev@bavard.ai",
    description="Machine learning model utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bavard-ai/bavard-ml-common",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "fastapi>=0.63.0",
        "pydantic>=1.7.3",
        "google-cloud-storage>=1.35.1",
        "google-cloud-pubsub>=2.2.0",
        "google-cloud-error-reporting>=1.1.1",
        "numpy>=1.15.4",
        "scikit-learn>=0.20.3",
        "requests>=2.21.0",
        "loguru>=0.5.1"
    ],
    cmdclass={"verify": VerifyVersionCommand},
)
