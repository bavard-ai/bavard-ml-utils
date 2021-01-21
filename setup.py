import os
import sys
import setuptools
from setuptools.command.install import install

# The version of this package
VERSION = "0.0.12"


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
    author="Bavard AI, LLC",
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
        "fastapi==0.61.1",
        "pydantic==1.7.2",
        "tensorflow==2.2.0",
        "google-cloud-storage==1.35.0",
        "google-cloud-pubsub==2.1.0",
        "protobuf==3.12.0",  # needed b/c of: https://github.com/googleapis/python-bigquery/issues/305
    ],
    cmdclass={"verify": VerifyVersionCommand},
)
