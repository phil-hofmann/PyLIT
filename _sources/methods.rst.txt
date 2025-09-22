.. _methods:

Methods
=======

.. toctree::
   :maxdepth: 1
   :hidden:

   methods/cdf_l2_fit
   methods/l1_reg
   methods/l2_fit
   methods/l2_reg
   methods/max_entropy_fit
   methods/tv_reg
   methods/var_reg

All method functions in **PyLIT** construct a least-squares functional of the form

.. math::
   :label: lsq-functional

   f(\boldsymbol{\alpha}) 
   = \tfrac{1}{2} \| R \boldsymbol{\alpha} - F \|_2^2 
   + \lambda r(\boldsymbol{\alpha}; D),

where

- :math:`\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_n)` are the coefficients.
- :math:`R` is the regression matrix,
- :math:`F` is the target vector,
- :math:`r(\boldsymbol{\alpha}; D, E)` is a method-specific regularization term,
- :math:`\lambda \ge 0` is the regularization parameter,
- :math:`D` is an optional default model.

Each method must return a :ref:`method` object that encapsulates that provides the name,
the objective function, its gradient, a closed-form or approximate solution 
(if available) and a learning rate estimate.

Template
--------

.. code-block:: python

   def method_template(
      megas: np.ndarray, D: np.ndarray, E: np.ndarray, lambd: FLOAT_DTYPE
   ) -> Method:
      r"""
      Template for a method function in PyLIT.

      This function performs type conversions, default handling,
      and delegates the actual specifications to the private function
      ``_method_template``.
      """

      # Type Conversion
      omegas = np.asarray(omegas).astype(FLOAT_DTYPE)
      D = np.asarray(D).astype(FLOAT_DTYP)
      E = np.asarray(E).astype(FLOAT_DTYPE)
      lambd = FLOAT_DTYPE(lambd)

      # Get method
      method = _method_template(omegas, D, E, lambd)

      # Compile
      _, m = E.shape
      x_, R_, F_, P_ = (
         np.zeros((m), dtype=FLOAT_DTYPE),
         np.eye(m, dtype=FLOAT_DTYPE),
         np.zeros((m), dtype=FLOAT_DTYPE),
         np.array([0], dtype=INT_DTYPE),
      )

      _ = method.f(x_, R_, F_)
      _ = method.grad_f(x_, R_, F_)
      _ = method.solution(R_, F_, P_)
      _ = method.lr(R_)

      return method

   def _optimizer_template(
      omegas, D, E, lambd
   ) -> np.ndarray:
      """
      Template for an optimization subroutine in PyLIT.
      """

      @njit(fastmath=FASTMATH)
      def f(x, R, F) -> FLOAT_DTYPE:
         raise NotImplementedError("This is only a template.")

      @njit(fastmath=FASTMATH)
      def grad_f(x, R, F) -> np.ndarray:
         raise NotImplementedError("This is only a template.")

      @njit(fastmath=FASTMATH)
      def solution(R, F, P) -> np.ndarray:
         raise NotImplementedError("This is only a template.")

      @njit(fastmath=FASTMATH)
      def lr(R) -> FLOAT_DTYPE:
         raise NotImplementedError("This is only a template.")

      return Method("method_template", f, grad_f, solution, lr)

Parameters
----------
- ``omegas`` (np.ndarray, optional):
   Discrete frequenciy axis (evaluation points).
   (default: ``None``).
- ``lambd`` (np.ndarray):
   Regularization parameter, determining the trade-off between fidelity and regularization.
- ``D`` (np.ndarray, optional):
    Default model vector, which is used in regularization terms.
- ``E`` (np.ndarray, optional):
    Evaluation matrix, which provides the kernel functions evaluated at :math:`\omega`.

Returns
-------
- ``Method``:
   A method object that implements the least-squares functional from :eq:`(*) <lsq-functional>`
   with optional regularization. The returned object is used to minimize the functional with 
   respect to the coefficients :math:`\boldsymbol{\alpha}`.
