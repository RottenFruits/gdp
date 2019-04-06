try:
    import setuptools
    from setuptools import setup, find_packages
except ImportError:
    print("Please install setuptools.")

import os
long_description = 'gdp is generating distributed representation code sets written by pytorch.'
if os.path.exists('README.MD'):
    long_description = open('README.MD').read()

setup(
    name  = 'gdp',
    version = '0.1',
    description = 'gdp is generating distributed representation code sets written by pytorch.',
    long_description = long_description,
    license = 'MIT',
    author = 'Shohei Ogawa',
    author_email = 's_ogawa@akarf.com',
    url = 'https://github.com/RottenFruits/gdp',
    keywords = 'distributed-representations',
    packages = find_packages(),
    install_requires = ['numpy', 'tqdm'],
    classifiers = [
      'Programming Language :: Python :: 3.6',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License'
    ]
)