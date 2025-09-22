Models
======

All models in this section are subclasses of the
:meth:`~pylit.models.lrm.LinearRegressionModel` class. 

Each model inherits the general linear regression framework and
requires only the implementation of two methods:

- **kernel** (:meth:`~pylit.models.lrm.LinearRegressionModel.kernel`):  
  Defines the model function for a given set of parameters and a discrete
  frequency axis.

- **ltransform** (:meth:`~pylit.models.lrm.LinearRegressionModel.ltransform`):  
  Defines the Laplace-transformed version of the model function, evaluated
  at the discrete time axis.
  
.. toctree::
   :maxdepth: 1
   :hidden:

   models/cauchy
   models/gauss
   models/laplace
   models/lrm
   models/uniform