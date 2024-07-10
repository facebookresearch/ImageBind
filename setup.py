from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='imagebind',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'imagebind': ['bpe/bpe_simple_vocab_16e6.txt.gz'],
    },
    description='A brief description of the package',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/facebookresearch/ImageBind',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International',
    ],
    install_requires=required,
    dependency_links=['https://download.pytorch.org/whl/cu113'],
)
