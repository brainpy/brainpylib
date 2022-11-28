Installation
============

.. contents::
    :local:
    :depth: 2


``brainpylib`` is designed to run cross platforms, including Windows,
GNU/Linux, and OSX.


Installation with pip
---------------------

You can install ``brainpylib`` from the `pypi <https://pypi.org/project/brainpylib/>`_.
To do so, use:

.. code-block:: bash

    pip install brainpylib

To update the latest version, you can use

.. code-block:: bash

    pip install -U brainpylib


If you want to install the pre-release version (the latest development version)
of ``brainpylib``, you can use:

.. code-block:: bash

   pip install --pre brainpylib



Installation from source
------------------------

If you decide not to use ``pip``, you can install ``brainpylib`` from
`GitHub <https://github.com/PKU-NIP-Lab/brainpylib>`_.

To do so, use:

.. code-block:: bash

    pip install git+https://github.com/PKU-NIP-Lab/brainpylib.git


Compile GPU operators
---------------------

We currently did not provide GPU wheel on Pypi
and users need to build ``brainpylib`` from source
if you want to GPU operators.
There are some prerequisites first:

- Linux platform.
- Nvidia GPU series required.
- `CUDA`_ and `CuDNN`_ have installed.

We have tested whole building process on Nvidia RTX A6000 GPU with CUDA 11.6 version.

First, obtain the source code:

.. code-block:: bash

    git clone https://github.com/PKU-NIP-Lab/brainpylib
    cd brainpylib

Then  compile GPU wheel via:

.. code-block:: bash

    python setup_cuda.py bdist_wheel

After compilation, it's convenient for users to install the package through following instructions:

.. code-block:: bash

    pip install dist/brainpylib-*.whl

``brainpylib-*.whl`` is the generated file from compilation, which is located in ``dist`` folder.

Now users have successfully install GPU version of ``brainpylib``,
and we recommend users to check if ``brainpylib`` can
be imported in the Python script.


.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _CuDNN: https://developer.nvidia.com/CUDNN
