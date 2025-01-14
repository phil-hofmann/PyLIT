1. **Implementing New Optimizer Methods**

To implement new methods in the provided structure, follow these steps:

(Exceptions: adaptive.py, solution.py)

1.1. **Import**

Import the required libraries at the beginning of the script:

```python
import pylit 
import numpy as np
import numba as nb
from pylit.optimize import Solution
# Further Imports

from pylit.global_settings import FLOAT_DTYPE, INT_DTYPE, ARRAY, TOL
```

1.2. **Function Signature**

Define the function signature with the required parameters and return type:

```python
def nn_doe(
    R: ARRAY,
    F: ARRAY,
    x0: ARRAY,
    method,
    maxiter=None,
    tol=None,
    protocol=False,
) -> Solution:
    """Solves the optimization problem using the non-negative method of John Doe."""
```

1.3. **Controlling**

Perform Integrity Checks and Type Conversion if necessary:

```python
# Integrity
if not R.shape[0] == len(F):
    raise ValueError(
        "The number of rows of R and the length of F must be equal"
    )

# Type Conversion
R = R.astype(FLOAT_DTYPE)
F = F.astype(FLOAT_DTYPE)
x0 = x0.astype(FLOAT_DTYPE)
```

1.4. **Defaults**

Set Default Values for Optional Parameters:

```python
# Prepare
n, m = R.shape
maxiter = 10 * n if maxiter is None else INT_DTYPE(maxiter)
tol = 10 * max(m, n) * TOL if tol is None else FLOAT_DTYPE(tol)
```

1.5. **Call Subroutine**

Call the subroutine to perform the main computation:

```python
# Subroutine
x = _nn_doe(
    R,
    F,
    method.f,
    method.grad_f,
    method.lr, # Optional
    method.solution, # Optional
    x0,
    maxiter,
    tol,
    method.pr,
    protocol,
)
fx = method.f(x, R, F)
```

1.6. **Solution**

Return the solution as a `Solution` object and the residuum:

```python
return Solution(x, fx, 0.5 * np.sum((R @ x - F) ** 2))
```

1.7. **Define Subroutine**

Define the subroutine to perform the main computation. Ensure to use Numba's JIT compilation for performance optimization:

```python
@nb.njit
def _nn_doe(
    R, F, f, grad_f, lr, solution, x0, maxiter, tol, pr, protocol
) -> ARRAY:
    # Initialize variables:
    x = np.copy(x0)

    # Subroutine implementation:

    return x
```