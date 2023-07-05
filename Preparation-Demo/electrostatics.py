import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import List


class ElectrostaticPotential:
    """Describing Potential from a System of Charges"""

    def __init__(self, charges: npt.ArrayLike[float], num_dimensions: int = 3) -> None:
        """Setup the system's initial configuration

        Parameters
        ==========
        charges : np.ArrayLike
            An array of charges to put in our system
            Shape should be (number_charges, num_dimensions + 1)
        num_dimensions : int
            The number of dimensions in the system
        """
        self.charges = charges

    @property
    def charges(self) -> npt.ArrayLike:
        """The charges in the system"""
        return self._charges

    @charges.setter
    def charges(self, charges) -> None:
        self._charges = charges

    @staticmethod
    def distance(x1: npt.ArrayLike, x2: npt.ArrayLike):
        """The Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def distance_to_charges(self, x1: npt.ArrayLike) -> npt.ArrayLike:
        """The distance to each charge

        Parameters
        ==========
        x1 : np.ArrayLike
            The point to compute the potential at
            Should have shape (num_dimensions, )

        Returns
        =======
        np.ArrayLike
            The array of distances
            should have shape (charges.shape[0],)
        """
        distances = np.zeros(self.charges.shape[0])
        distances = np.apply_along_axis(self.distance, 1, self.charges[:, 1:], x1)
        return distances

    def potential_at_point(self, x1: npt.ArrayLike[float]) -> float:
        """The electrostatics potential at a point

        Parameters
        ==========
        x1 : np.ArrayLike
            The point at which to compute the potential

        Returns
        =======
        float
            The potential at this point
        """
        distances = self.distance_to_charges(x1)
        chargewise_potential = -self.charges[:, 0] / 4 / np.pi / (8.854e-12) / distances
        potential = np.sum(chargewise_potential)
        return potential


if __name__ == "__main__":
    num_charges = 10000000
    charges = np.zeros((num_charges, 4))
    charges[:, 0] = np.random.normal(scale=1.6e-19, size=num_charges)
    from scipy.stats import multivariate_normal

    positions_generator = multivariate_normal(mean=np.zeros(3), cov=np.identity(3))
    charges[:, 1:] = positions_generator.rvs(num_charges)
    Test = ElectrostaticPotential(charges)
    print(Test.potential_at_point(np.zeros(3)))
