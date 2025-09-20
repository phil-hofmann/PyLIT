Anaconda
========

To include this package in your project using **Anaconda**, follow these steps:

1. **Ensure that Anaconda or Miniconda is installed**
You can check with:

.. code-block:: bash

    conda --version

2. **Create a Conda environment with Python 3.12 or newer (if not already set up)**

.. code-block:: bash

    conda create --name venv python=3.12

3. **Activate the environment**

.. code-block:: bash

    conda activate venv

4. **Ensure that pip is installed (it should be by default)**

.. code-block:: bash

    conda install pip && pip --version

5. **Use pip to install pylit directly from the GitHub repository**

.. code-block:: bash

    pip install git+https://github.com/phil-hofmann/pylit.git

**(Optional) Install from a specific branch**

.. code-block:: bash

    pip install git+https://github.com/phil-hofmann/pylit.git@branch_name


6. **To verify that pylit was installed correctly, you can run**

.. code-block:: bash

    pip show pylit

7. **When you're done working, deactivate the environment**

.. code-block:: bash

    conda deactivate