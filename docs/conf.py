# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# a_list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

import brainpylib
import shutil
from docs import auto_generater

base_path = 'apis/auto/'
if not os.path.exists(base_path):
  os.makedirs(base_path)

shutil.copyfile('../changelog.rst', os.path.join(base_path, 'changelog.rst'))

auto_generater.write_module(module_name='brainpylib.event_ops',
                            filename=os.path.join(base_path, 'event_ops.rst'),
                            header='Event-driven computation operators')

auto_generater.write_module(module_name='brainpylib.sparse_ops',
                            filename=os.path.join(base_path, 'sparse_ops.rst'),
                            header='Sparse computation operators')

auto_generater.write_module(module_name='brainpylib.jitconn_ops',
                            filename=os.path.join(base_path, 'jitconn_ops.rst'),
                            header='Just-in-time connectivity operators')

auto_generater.write_module(module_name='brainpylib.op_register',
                            filename=os.path.join(base_path, 'op_register.rst'),
                            header='Operator registration routines')

auto_generater.write_module(module_name='brainpylib.compat',
                            filename=os.path.join(base_path, 'compat.rst'),
                            header='Old compatible operators')

# -- Project information -----------------------------------------------------

project = 'brainpylib'
copyright = '2022, brainpylib'
author = 'BrainPy Team'

# The full version, including alpha/beta/rc tags
release = brainpylib.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.intersphinx',
  'sphinx.ext.mathjax',
  # 'sphinx-mathjax-offline',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'sphinx_autodoc_typehints',
  'myst_nb',
  # 'matplotlib.sphinxext.plot_directive',
  'sphinx_thebe'
]
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# source_suffix = '.rst'
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

intersphinx_mapping = {
  "python": ("https://docs.python.org/3.8", None),
  "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}
nitpick_ignore = [
  ("py:class", "docutils.nodes.document"),
  ("py:class", "docutils.parsers.rst.directives.body.Sidebar"),
]

suppress_warnings = ["myst.domains", "ref.ref"]

numfig = True

myst_enable_extensions = [
  "dollarmath",
  "amsmath",
  "deflist",
  # "html_admonition",
  # "html_image",
  "colon_fence",
  # "smartquotes",
  # "replacements",
  # "linkify",
  # "substitution",
]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "sphinx_book_theme"
html_logo = "_static/logo.png"
html_title = "brainpylib documentation"
html_copy_source = True
html_sourcelink_suffix = ""
html_favicon = "_static/logo-square.png"
html_last_updated_fmt = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
jupyter_execute_notebooks = "off"
thebe_config = {
  "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
  "repository_branch": "master",
}

html_theme_options = {
  'logo_only': True,
  'show_toc_level': 2,
}

# -- Options for myst ----------------------------------------------
# Notebook cell execution timeout; defaults to 30.
execution_timeout = 200
