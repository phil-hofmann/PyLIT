Installation
============

- :ref:`install-poetry`
- :ref:`install-anaconda`
- :ref:`install-local`

.. _install-poetry:

Including in Your Project with Poetry
-------------------------------------

To include this package in your project using **Poetry**, follow these steps:

1. **Ensure pip is installed**

.. code-block:: bash
    
    python3 -m ensurepip --upgrade

2. **Install Poetry if not already installed**

.. code-block:: bash

    pip install --upgrade poetry
    poetry --version

3. **(Optional) Initialize pyproject.toml if you don’t have one**

.. code-block:: bash

    poetry init


4. **Install pylit directly from GitHub**

.. code-block:: bash

    poetry add git+https://github.com/phil-hofmann/pylit.git

**(Optional) Install from a specific branch**

.. code-block:: bash

    poetry add git+https://github.com/phil-hofmann/pylit.git@branch_name

5. **Verify installation**

.. code-block:: bash

    poetry show pylit

6. **Run your code inside the Poetry environment**

.. code-block:: bash

    poetry run python my_python_file.py

.. _install-anaconda:

Including in Your Project with Anaconda
---------------------------------------

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


.. _install-local:

Set Up the Repository on Your Local Machine
-------------------------------------------

1. **Clone the repository**

.. code-block:: bash

    git clone https://github.com/phil-hofmann/pylit.git
    cd pylit

2. **Ensure pip is installed**

.. code-block:: bash

    python3 -m ensurepip --upgrade

3. **Install Poetry if not already installed**

.. code-block:: bash

    pip install --upgrade poetry
    poetry --version

4. **Install dependencies using Poetry**

.. code-block:: bash

    poetry install