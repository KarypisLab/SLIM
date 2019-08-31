#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:12:37 2019

@author: shuix007
"""

import os
import platform
import site
from setuptools import setup

PLATFORM_SYSTEM = 'Linux'
EXTENSION = 'so'
if platform.system().startswith('Darwin'):
    PLATFORM_SYSTEM = 'Darwin'
    EXTENSION = 'dylib'

SHARED_LIBRARY_PATH = '../build/' + PLATFORM_SYSTEM + \
                        '-x86_64/src/libslim/libslim.' + EXTENSION

setup(
    name='SLIM',
    version='2.0.0',
    author=
    'George Karypis, Athanasios N. Nikolakopoulos, Xia Ning, Mohit Sharma, Zeren Shui',
    author_email='karypis@cs.umn.edu',
    install_requires=['numpy', 'scipy'],
    packages=['SLIM'],
    url='https://github.umn.edu/dminers/slim.git',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ])

site_path = site.getsitepackages()[0]
if not os.path.isdir(site_path + '/SLIM'):
    os.mkdir(site_path + '/SLIM')
print('site_path: ', site_path)
os.system(' '.join(['cp', SHARED_LIBRARY_PATH,
                    site_path + '/SLIM/libslim.so']))
