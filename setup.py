# -*- coding: utf-8 -*-

import io
import os
import re
import glob
import sys

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
if sys.platform == 'darwin': # mac
  ext_modules = [
    Pybind11Extension("brainpylib/cpu_ops",
                      sources=glob.glob("lib/cpu_*.cc") + glob.glob("lib/cpu_*.cpp"),
                      cxx_std=11,
                      extra_link_args=["-rpath", re.sub('/lib/.*', '/lib', sys.path[1])],
                      define_macros=[('VERSION_INFO', __version__)]),
  ]
else:
  ext_modules = [
    Pybind11Extension("brainpylib/cpu_ops",
                      sources=glob.glob("lib/cpu_*.cc") + glob.glob("lib/cpu_*.cpp"),
                      cxx_std=11,
                      define_macros=[('VERSION_INFO', __version__)]),
  ]


# obtain long description from README
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
  README = f.read()

# build
setup(
  name='brainpylib',
  version=__version__,
  description='C++/CUDA Library for BrainPy',
  long_description=README,
  long_description_content_type="text/markdown",
  author='BrainPy team',
  author_email='chao.brain@qq.com',
  packages=find_packages(exclude=['lib*', 'docs', 'tests']),
  include_package_data=True,
  install_requires=["jax", "jaxlib", "numba", "numpy"],
  extras_require={"test": "pytest"},
  python_requires='>=3.7',
  url='https://github.com/brainpy/brainpylib',
  ext_modules=ext_modules,
  cmdclass={"build_ext": build_ext},
  license='GPL-3.0 license',
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
