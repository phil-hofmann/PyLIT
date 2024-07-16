from pylit.backend.utils import to_string

class Method: # TODO put to dataclasses...
    """Represents a method for solving an optimization problem."""

    def __init__(
        self,
        name: str,
        f: callable,
        grad_f: callable,
        solution: callable,
        lr: callable,
        pr: callable,
    ):
        self.name = name
        self.f = f
        self.grad_f = grad_f
        self.solution = solution
        self.lr = lr
        self.pr = pr

    @property
    def name(self) -> str:
        """Name of the method."""
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Name must be a string.")
        self._name = value

    @property
    def f(self) -> callable:
        """Objective function."""
        return self._f

    @f.setter
    def f(self, value: callable):
        if not callable(value):
            raise TypeError("f must be a callable function.")
        self._f = value

    @property
    def grad_f(self) -> callable:
        """Gradient of the objective function."""
        return self._grad_f

    @grad_f.setter
    def grad_f(self, value: callable):
        if not callable(value):
            raise TypeError("grad_f must be a callable function.")
        self._grad_f = value

    @property
    def solution(self) -> callable:
        """Solution."""
        return self._solution

    @solution.setter
    def solution(self, value: callable):
        if not callable(value):
            raise TypeError("solution must be a callable function.")
        self._solution = value

    @property
    def lr(self) -> callable:
        """Learning rate function."""
        return self._lr

    @lr.setter
    def lr(self, value: callable):
        if not callable(value):
            raise TypeError("lr must be a callable function.")
        self._lr = value

    @property
    def pr(self) -> callable:
        """Projection operator."""
        return self._pr

    @pr.setter
    def pr(self, value: callable):
        if not callable(value) and value is not None:
            raise TypeError("pr must be a callable function or None.")
        self._pr = value

    def __str__(self) -> str:
        """Class string method.

        Returns
        -------
        str
            The string representation of the class.
        """

        return to_string(self)

if __name__ == "__main__":
    pass