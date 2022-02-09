from distutils.core import setup
from setuptools import find_packages

def readme():
    try:
        with open('README.rst') as f:
            return f.read()
    except:
        pass

setup(
    name = 'padasip',
    packages = find_packages(exclude=("tests",)),
    version = '1.2.1',
    description = 'Python Adaptive Signal Processing',
    long_description = readme(),
    author = 'Matous Cejnek',
    maintainer = "Matous Cejnek",
    author_email = 'matousc@gmail.com',
    license = 'MIT',
    url = 'http://matousc89.github.io/padasip/',
    download_url = 'https://github.com/matousc89/padasip/',
    keywords = ['signal-processing', 'adaptive filters'],
    install_requires=[
        'numpy',
    ],
    bugtrack_url = "https://github.com/matousc89/padasip/issues",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Adaptive Technologies',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Programming Language :: Python',
    ],
)
