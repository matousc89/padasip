from distutils.core import setup

def readme():
    try:
        with open('README.rst') as f:
            return f.read()
    except:
        pass

setup(
    name = 'padasip',
    packages = ['padasip'],
    version = '0.1',
    description = 'Python Adaptive Signal Processing',
    long_description = readme(),
    author = 'Matous Cejnek',
    author_email = 'matousc@gmail.com',
    license = 'MIT',
    url = 'https://github.com/matousc89/padasip.git',
    download_url = 'https://github.com/matousc89/padasip.git/tarball/0.1',
    keywords = ['adaptive', 'signal-processing'],
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
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
