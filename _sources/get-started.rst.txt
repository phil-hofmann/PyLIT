Get Started
===========

**Before running** any example, you must first follow the installation guide described in :doc:`installation`.  

Two example configurations are provided in the ``PyLIT/examples/`` folder which you can
download `here <https://github.com/phil-hofmann/PyLIT/archive/refs/heads/main.zip>`_.

- **Beginner**

  A minimal setup with synthetic test data, designed to help you quickly verify that the 
  installation was successful and to introduce you to the basic workflow of PyLIT.

  Once installed, you can run the beginner example by executing the provided ``run.py`` script.  
  For instance, if you have set up your environment with Poetry, you can run:

  .. code-block:: bash

    poetry run python run.py
  
  This example is intentionally simple, so you can focus on understanding the fundamental steps before moving on to more advanced features.  
  It demonstrates the **core workflow of PyLIT**:

  1. Define the required input and output paths.  
  2. Instantiate a **configuration object**, which provides access to a wide range of options for controlling the inversion process.  
  3. Call **``prepare``** to load and preprocess the input CSV files (``F.csv`` and ``D.csv``).  
  4. After preparation, perform the inversion using **``itransform``** to obtain the double-sided inverse Laplace transform. 

- **ESA synthetic data**  

  A more detailed example that uses realistic synthetic data.  
  This setup is intended for users who want to explore more advanced features, experiment with tuning options, 
  and study how priors affect the inversion.

  To run this example, you only need to make the provided bash script executable and then execute it:

  .. code-block:: bash

     chmod +x run_sample_usage_files.sh
     ./run_sample_usage_files.sh

  If you are working with Anaconda, you may need to adapt the script to your environment.


Note
----

PyLIT **always** requires two CSV files to run

- **F.csv**: contains the imaginary-time correlation function :math:`F(q, \tau)` 
  evaluated at a discrete time axis :math:`\tau`.  
  The **first column** must be the discrete time axis :math:`\tau`, and the **subsequent columns** 
  contain :math:`F(q, \tau)` for each q-point. 
- **D.csv**: contains the default model :math:`D(q, \omega)` for the 
  dynamic structure factor :math:`S(q, \omega)`, evaluated at a discrete frequency axis :math:`\omega`.  
  The **first column** must be the discrete frequency axis :math:`\omega` values, and the **subsequent columns** 
  contain :math:`S(q, \omega)` for each q-point.  
  This serves as a prior estimate to guide the Laplace inversion.