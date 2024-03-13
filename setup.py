from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="imagebind-packaged",
    version="0.1.2",
    author='Raghav Dixit',
    packages=find_packages(),
    include_package_data=True,
    description="Updated version of Imagebind package with bug fixes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/raghavdixit99/ImageBind",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=required,
    dependency_links=["https://download.pytorch.org/whl/cu113"],
)
