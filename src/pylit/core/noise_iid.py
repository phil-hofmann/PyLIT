import numpy as np
from abc import ABC, abstractmethod
from pylit.global_settings import INT_DTYPE, ARRAY, FLOAT_DTYPE

class NoiseIID(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def sample(self, size: INT_DTYPE) -> ARRAY:
        """ Sample from the noise distribution. """
        pass

    def __call__(self, x: ARRAY) -> ARRAY:
        """ Add noise to the input data. """
        return x + self.sample(len(x))
    
class WhiteNoise(NoiseIID):
        
    def __init__(self, mean: FLOAT_DTYPE, std: FLOAT_DTYPE):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def sample(self, size: INT_DTYPE) -> ARRAY:
        return np.random.normal(self.mean, self.std, size) 

class UniformNoise(NoiseIID):
        
    def __init__(self,low: FLOAT_DTYPE, high: FLOAT_DTYPE):
        super().__init__()
        self.low = low
        self.high = high
        
    def sample(self, size: INT_DTYPE) -> ARRAY:
        return np.random.uniform(self.low, self.high, size)

class BernoulliNoise(NoiseIID):
        
    def __init__(self, prob: FLOAT_DTYPE):
        super().__init__()
        self.prob = prob
        
    def sample(self, size: INT_DTYPE) -> ARRAY:
        return np.random.binomial(n=1, p=self.prob, size=size)

class PoissonNoise(NoiseIID):
        
    def __init__(self, lam: FLOAT_DTYPE):
        super().__init__()
        self.lam = lam
        
    def sample(self, size: INT_DTYPE) -> ARRAY:
        return np.random.poisson(self.lam, size)

# Add more noise types as needed
