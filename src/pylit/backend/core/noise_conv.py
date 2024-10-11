import numpy as np
from scipy.stats import binom
from abc import ABC, abstractmethod
from pylit.global_settings import INT_DTYPE, ARRAY, FLOAT_DTYPE

# NOTE Add more convolutions as needed


class ConvolutionKernel(ABC):

    def __init__(self, window: INT_DTYPE):
        self.window = window

    @abstractmethod
    def kernel(self) -> ARRAY:
        """Define the kernel."""
        pass

    def __call__(self, x: ARRAY) -> ARRAY:
        """Apply the kernel to the input data."""
        x = np.asarray(x).astype(FLOAT_DTYPE)
        if x.shape[0] < self.window:
            raise ValueError("Input data must be at least as long as the window size.")
        if x.ndim == 1:
            return np.convolve(x, self.kernel(), "same")
        elif x.ndim == 2:
            return np.array(
                [np.convolve(x[i], self.kernel(), "same") for i in range(x.shape[0])]
            )
        else:
            raise ValueError("Only 1D and 2D data are supported.")


class BinomKernel(ConvolutionKernel):

    def __init__(self, window: INT_DTYPE, prob: FLOAT_DTYPE):
        super().__init__(window)
        self.prob = prob

    def kernel(self) -> ARRAY:
        return np.array(
            [binom.pmf(k, self.window - 1, self.prob) for k in range(self.window)],
            dtype=FLOAT_DTYPE,
        )


class UniformKernel(ConvolutionKernel):

    def __init__(self):
        super().__init__()

    def kernel(self) -> ARRAY:
        return np.ones(self.window) / self.window
