# -*- coding: utf-8 -*-

import os
import re
import glob

from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext

build_ext.get_export_symbols = lambda *args: []

# version control
HERE = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(HERE, 'brainpylib', '__init__.py'), 'r') as f:
  init_py = f.read()
  __version__ = re.search('__version__ = "(.*)"', init_py).groups()[0]

# extension modules
ext_modules = [
  Pybind11Extension("brainpylib/cpu_ops",
                    sources=glob.glob("lib/cpu_*.cc"),
                    cxx_std=11,
                    define_macros=[('VERSION_INFO', __version__)]),
]


# build
setup(
  name='brainpylib',
  version=__version__,
  description='C++/CUDA Library for BrainPy',
  author='BrainPy team',
  author_email='chao.brain@qq.com',
  packages=find_packages(exclude=['lib*']),
  include_package_data=True,
  install_requires=["jax", "jaxlib", "pybind11>=2.6", "cffi", "numba"],
  extras_require={"test": "pytest"},
  python_requires='>=3.7',
  url='https://github.com/PKU-NIP-Lab/brainpylib',
  ext_modules=ext_modules,
  cmdclass={"build_ext": build_ext},
  license='Apache-2.0 License',
  keywords=('event-driven computation, '
            'sparse computation, '
            'brainpy'),
  classifiers=[
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ],
)
