Poetry
======

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