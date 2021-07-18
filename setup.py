from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Joycelyn Longdon",
    description="AI4ER Masters Research Project",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
