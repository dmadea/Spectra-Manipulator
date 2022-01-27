from setuptools import setup, find_packages
from spectramanipulator import __version__

DESCRIPTION = 'Spectra Manipulator based on pyqtgraph'

setup(
    name="spectramanipulator",
    version=__version__,
    author="Dominik Madea",
    author_email="dominik.madea@gmail.com",
    url="https://github.com/dmadea/Spectra-Manipulator",
    download_url=f"https://github.com/dmadea/Spectra-Manipulator/archive/refs/tags/v{__version__}.tar.gz",
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    packages=find_packages(),
    python_requires='>3.7.1',
    install_requires=[
        'lmfit',  # installs numpy and scipy
        'numpy',
        'scipy',
        'pyqtgraph>=0.12.1',
        'matplotlib',
        'pyqt5',
        'qtconsole>=5.1.0',
        'numba',
        'requests',
        'winshell',
        'cmlib'
    ],

    keywords=['python', 'spectra manipulator'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English"
    ]
)