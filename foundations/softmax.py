import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        num = np.exp((z - max(z)))
        den = np.sum(num)
        softmax = num/den
        return np.round(softmax,4)
