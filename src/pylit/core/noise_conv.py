import numpy as np
from scipy.stats import binom
from abc import ABC, abstractmethod
from pylit.global_settings import INT_DTYPE, ARRAY, FLOAT_DTYPE

class ConvolutionKernel(ABC):
    
    def __init__(self, window: INT_DTYPE):
        self.window = window
    
    @abstractmethod
    def kernel(self, size: INT_DTYPE) -> ARRAY:
        """ Define the kernel. """
        pass

    def __call__(self, x: ARRAY) -> ARRAY:
        """ Apply the kernel to the input data. """
        return np.convolve(x, self.kernel(), 'same')
    
class BinomKernel(ConvolutionKernel):
        
    def __init__(self, window: INT_DTYPE, prob: FLOAT_DTYPE):
        super().__init__(window)
        self.prob = prob
        
    def kernel(self) -> ARRAY:
        return [binom.pmf(k, self.window-1, self.prob) for k in range(self.window)]

class UniformKernel(ConvolutionKernel):
        
    def __init__(self):
        super().__init__()
        
    def kernel(self) -> ARRAY:
        return np.ones(self.window) / self.window

# Add more convolutions as needed