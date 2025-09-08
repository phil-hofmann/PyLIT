.. _optimizer:

Optimizer
=========

.. toctree::
   :maxdepth: 1
   
   optimizer/adam
   optimizer/adaptive
   optimizer/nesterov
   optimizer/nnls

Interface
---------
All optimization functions in PyLIT solve the constrained least-squares optimization problem

    .. math::
      :label: lsq-problem

      \min_{x \geq 0} \; \frac{1}{2} \| R x - F \|^2

iteratively and *must follow a consistent interface*.
The following template specifies how an optimizer function should look, including the required inputs, optional parameters, and expected output.

Template
^^^^^^^^

.. code-block:: python

    def optimizer_template(
        R: np.ndarray,
        F: np.ndarray,
        x0: np.ndarray,
        method: Method,
        maxiter: INT_DTYPE = None,
        tol: FLOAT_DTYPE = None,
        svd: bool = False,
        protocol: bool = False,
    ) -> Solution:
      """
      Template for an optimization function in PyLIT.

      This function performs type conversions, default handling, 
      and delegates the actual optimization to the private function ``_optimizer_template``.
      """

      # Convert types
      R = R.astype(FLOAT_DTYPE)
      F = F.astype(FLOAT_DTYPE)
      x0 = x0.astype(FLOAT_DTYPE)

      # Handle defaults
      n, m = R.shape
      maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
      tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)

      # Call subroutine
      x = _optimizer_template(
        R,
        F,
        method.f,
        method.grad_f,
        x0,
        method.lr,
        maxiter,
        tol,
        protocol,
        svd,
      )

      # Compute objective
      fx = method.f(x, R, F)

      return Solution(x, fx, FLOAT_DTYPE(0.5 * np.sum((R @ x - F) ** 2)))


      def _optimizer_template(
         R, F, f, grad_f, x0, lr, maxiter, tol, protocol, svd
         ) -> np.ndarray:
        """
        Template for an optimization subroutine in PyLIT.
        """
        raise NotImplementedError("This is only a template.")

Parameters
^^^^^^^^^^

- ``R`` (np.ndarray):
   Matrix of shape ``(m, n)`` representing the system to solve.

- ``F`` (np.ndarray):
   Data vector of shape ``(m,)``.

- ``x0`` (np.ndarray):
   Initial guess for the solution, a vector of shape ``(n,)``.

- ``method`` (Method):
   Configuration object specifying algorithm-specific parameters.

- ``maxiter`` (INT_DTYPE, optional):
   Maximum number of iterations.
   (default: ``10 * n``).

- ``tol`` (FLOAT_DTYPE, optional): 
   Convergence tolerance. Iterations stop when the change in the objective function falls below this value. 
   (default: ``10 * max(m, n) * TOL``).

- ``svd`` (bool, optional):
   If ``True``, apply SVD-based preprocessing to stabilize the optimization. 
   (default: ``False``).

- ``protocol`` (bool, optional):
   If ``True``, prints an iteration protocol showing the current step number and corresponding error.
   (default: ``False``).

Returns
^^^^^^^

- ``Solution``:
   The Solution object containing the final iterate ``x``, the objective function value ``eps``,
   and the residuum ``res`` from :eq:`(*) <lsq-problem>`.

Notes
^^^^^

- This template serves as a **blueprint**: it defines how optimizer functions
  **need** to be structured. Actual implementations (e.g., ADAM, Nesterov) 
  follow this signature.
- Using a consistent template ensures that all optimizers in PyLIT can be
  used interchangeably.
