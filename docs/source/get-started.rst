Get Started
===========

Two example configurations are provided in the ``get-started/`` folder which you can access `here <_static/get-started/>`_.

- **Beginner**

  A minimal setup with synthetic test data.  
  Useful to quickly test the installation and understand the basic workflow.  

- **ESA synthetic data**  

  A more detailed example using realistic synthetic data.
  Useful to explore more features, tuning options, and the effect of priors.


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