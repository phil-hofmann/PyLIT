import numpy as np
from abc import ABC, abstractmethod
from pylit.global_settings import INT_DTYPE, ARRAY, FLOAT_DTYPE

# NOTE
# - Only specify the Types, frontend will specify the rest through the PARAM_MAP
# - Symmetric noise only works for linear tau-spaces
# - Add more noise types as needed at the end


# Utilities
def symmetric_noise(noise: ARRAY) -> ARRAY:
    n = len(noise)
    n_half = n // 2
    noise_half = noise[:n_half]
    if n % 2 == 0:
        return np.concatenate((noise_half, noise_half[::-1]))
    else:
        return np.concatenate(
            (
                noise_half,
                np.array([noise[n_half + 1]], dtype=FLOAT_DTYPE),
                noise_half[::-1],
            )
        )


# Noise Classes
class NoiseIID(ABC):

    def __init__(self, sample_size: INT_DTYPE):
        self.sample_size = sample_size

    @abstractmethod
    def sample(self, size: INT_DTYPE) -> ARRAY:
        """Sample one realization from the noise distribution."""
        pass

    def __call__(self, size: INT_DTYPE) -> ARRAY:
        """Generate noise samples."""
        return [self.sample(size) for _ in range(self.sample_size)]


class WhiteNoise(NoiseIID):

    def __init__(
        self,
        mean: FLOAT_DTYPE,
        std: FLOAT_DTYPE,
        sample_size: INT_DTYPE,
        symmetric: bool,
    ):
        super().__init__(sample_size)
        self.mean = mean
        self.std = std
        self.symmetric = symmetric

    def sample(self, size: INT_DTYPE) -> ARRAY:
        noise = np.random.normal(self.mean, self.std, size)
        if self.symmetric:
            return symmetric_noise(noise)
        return noise


class UniformNoise(NoiseIID):

    def __init__(
        self,
        low: FLOAT_DTYPE,
        high: FLOAT_DTYPE,
        sample_size: INT_DTYPE,
        symmetric: bool,
    ):
        super().__init__(sample_size)
        self.low = low
        self.high = high
        self.symmetric = symmetric

    def sample(self, size: INT_DTYPE) -> ARRAY:
        noise = np.random.uniform(self.low, self.high, size)
        if self.symmetric:
            return symmetric_noise(noise)
        return noise


class BernoulliNoise(NoiseIID):

    def __init__(
        self,
        prob: FLOAT_DTYPE,
        sample_size: INT_DTYPE,
        symmetric: bool,
    ):
        super().__init__(sample_size)
        self.prob = prob
        self.symmetric = symmetric

    def sample(self, size: INT_DTYPE) -> ARRAY:
        noise = np.random.binomial(n=1, p=self.prob, size=size)
        if self.symmetric:
            return symmetric_noise(noise)
        return noise


class PoissonNoise(NoiseIID):

    def __init__(
        self,
        lam: FLOAT_DTYPE,
        sample_size: INT_DTYPE,
        symmetric: bool,
    ):
        super().__init__(sample_size)
        self.lam = lam
        self.symmetric = symmetric

    def sample(self, size: INT_DTYPE) -> ARRAY:
        noise = np.random.poisson(self.lam, size)
        if self.symmetric:
            return symmetric_noise(noise)
        return noise
