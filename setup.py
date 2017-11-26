# -*- coding: utf-8 -*-
from distutils.core import setup


version = '1.0.5'

setup(
    name = 'mapsplotlib',
    packages = ['mapsplotlib'],
    version = version,
    description = 'Custom Python plots on a Google Maps background. A flexible matplotlib like interface to generate many types of plots on top of Google Maps.',
    url = 'https://github.com/tcassou/mapsplotlib',
    download_url = 'https://github.com/tcassou/mapsplotlib/archive/{}.tar.gz'.format(version),
    keywords = ['google', 'maps', 'matplotlib', 'python', 'plot'],
    classifiers = [
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    install_requires=[
        'numpy>=1.8.2',
        'pandas>=0.13.1',
        'scipy>=0.13.3',
        'matplotlib>=1.3.1',
        'requests>=2.18.4',
        'pillow>=4.3.0',
    ],
)
