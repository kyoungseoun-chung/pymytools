#!/usr/bin/env python3
"""Random number generators."""
from dataclasses import dataclass
from math import cos
from math import sin
from math import sqrt

import torch
from scipy.stats import pearson3
from torch import Tensor

from pymytools.constants import PI


@dataclass
class Randoms:
    """Class for random number generators. Created as separate class to set
    fixed random number seed but not working as I intended yet.

    Note:
        - Suffix with `_v` means vector form, `_t` means tensor form.
    """

    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")
    seed: int | None = None

    def __post_init__(self):
        """You can specify seed to have fixed pattern of random numbers.

        Args:
            seed_in (int, optional): random seed. Defaults to None.
        """

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def uniform(self) -> float:
        """rand_uniform generate a uniform random number [0 1)

        Returns:
            float: np.random.rand()
        """
        return torch.rand(dtype=self.dtype, device=self.device).item()

    def normal(self) -> float:
        """Normal random number."""
        return torch.randn(dtype=self.dtype, device=self.device).item()

    def normal_v(self, n: int) -> Tensor:
        """Normal random numbers in vector form."""
        return torch.randn(n, dtype=self.dtype, device=self.device)

    def normal_t(self, n_row: int, n_col: int) -> Tensor:
        """Normal random number tensor.

        Args:
            n_row (int): number of rows of output
            n_col (int): number of columns of output

        Returns:
            Tensor
        """
        return torch.randn((n_row, n_col), dtype=self.dtype, device=self.device)

    def uniform_v(self, n: int) -> Tensor:
        """Uniform random numbers in vector form."""
        return torch.rand(n, dtype=self.dtype, device=self.device)

    def uniform_t(self, n_row: int, n_col: int) -> Tensor:
        """Uniform random number tensor.

        Args:
            n_row (int): number of rows of output
            n_col (int): number of columns of output

        Returns:
            Tensor
        """
        return torch.rand((n_row, n_col), dtype=self.dtype, device=self.device)

    def bi_normal_v(self, mean: list, std: list, n: int) -> Tensor:
        """random bi-modal normal distribution

        Note:
            here, the mean and std are for individual normal distribution
            to construct bi-modal normal distribution. Resulting distribution
            is scaled to have zero mean and unity standard deviation
        """

        rnd_bi_norm = self.skew_normal_t([0, 0], mean, std, n, 2)

        rnd = rnd_bi_norm.sum(dim=1)
        rnd -= rnd.mean()
        rnd /= rnd.std()

        return rnd

    def skew_normal_t(
        self, alpha: list, mean: list, std: list, n: int, m: int
    ) -> Tensor:
        """3d skewed normal distribution

        Note:
            if alpha is zero, it returns the Gaussian distribution

        Args:
            alpha (list): shape factors
            mean (list): mean of distribution
            std (list): standard deviation of distribution
            n (int): number of samples
            m (int): resired dimension of the system

        Returns:
            ndarray: m x pearson.rvs(skew, loc, scale, n)

        """

        rnd = torch.zeros((n, m), dtype=self.dtype, device=self.device)

        for i in range(m):
            rnd[:, i] = self.skew_normal_v(alpha[i], mean[i], std[i], n)

        return rnd

    def skew_normal_v(
        self, alpha: float, mean: float, std: float, n_samples: int
    ) -> Tensor:
        """rand_skew_normal creates numpy array filled with
        skew-normal-random numbers.
        If alpha is set to 'zero' then it will give normal random number.

        Args:
            alpha (float): shape factors
            mean (float): mean of distribution
            std (float): standard deviation of distribution
            n_samples (int): number of samples

        Returns:
            Tensor: skew-normal-random numbers
        """
        delta = alpha / sqrt(1 + pow(alpha, 2))
        scale = std / sqrt(1 - 2 * pow(delta, 2) / PI)
        loc = mean - scale * delta * sqrt(2 / PI)
        skew = ((4 - PI) / 2) * (
            pow(delta * sqrt(2 / PI), 3) / pow(1 - 2 * pow(delta, 2) / PI, 1.5)
        )

        return torch.from_numpy(pearson3.rvs(skew, loc, scale, n_samples)).to(
            dtype=self.dtype, device=self.device
        )

    def maxwell_v(
        self, n_pt: int, loc: float = 0.0, beta: float = 1.0 / sqrt(2.0)
    ) -> Tensor:
        """Rnadom Maxwell distribution."""
        return (
            torch.sqrt(
                -torch.log(torch.rand(n_pt, dtype=self.dtype, device=self.device))
            )
            / beta
            + loc
        )

    def from_maxed_v(
        self,
        coeffs: Tensor,
        pdf: Tensor,
        bounds: list[float] | tuple[float, float],
        n_pt: int,
    ) -> Tensor:
        """Sample n_pt number of random varables according to the Maximum
        Entropy Distribution (MED) by the given coefficients.

        Note:
            Sampling is done by the acceptance-rejection method in 1 dimension

        Args:
            bounds (tuple or list): lower and upper bounds of the random
                variable's domain
            n_pt (int): number of random variables to be sampled

        Returns:
            ndarray: sampled random variables
        """
        w_bounds = bounds[1] - bounds[0]
        h_bounds = bounds[1]

        rnd_sample = _rejection_sampling(w_bounds, h_bounds, coeffs, pdf, n_pt)

        return rnd_sample

    def cylinder_t(
        self,
        n_pt: int,
        bounds: list[list[float]] | tuple[tuple[float, float], ...] = ((0, 5), (-5, 5)),
    ) -> Tensor:
        r"""Uniform grid sampling in the cylindrical coordinates.

        Note:
            .. math::
                r &= \sqrt{(x^2 + y^2)} \\
                z &= z

        Args:
            n_pt (int): number of samples
            bounds (tuple): min-max boundary in cylindrical coordinates

        Returns:
            Tensor: size of n_pt x 3
        """

        r_min = bounds[0][0]
        r_max = bounds[0][1]

        z_min = bounds[1][0]
        z_max = bounds[1][1]

        rnd_r = torch.rand(n_pt, dtype=self.dtype, device=self.device)
        rnd_z = torch.rand(n_pt, dtype=self.dtype, device=self.device)
        rnd_theta = 2.0 * PI * torch.rand(n_pt, dtype=self.dtype, device=self.device)

        r = (r_max - r_min) * rnd_r + r_min

        x = r * cos(rnd_theta)
        y = r * sin(rnd_theta)
        z = (z_max - z_min) * rnd_z + z_min

        return torch.vstack((x, y, z)).T


def _rejection_sampling(
    w_bounds: float, h_bounds: float, coeffs: Tensor, pdf: Tensor, n_pt: int
) -> Tensor:
    """Sample random number from the pdf using the acceptance-rejection method."""
    rnd_sample = torch.zeros(n_pt)
    i_sample = 0

    while i_sample < n_pt:
        loc = w_bounds * torch.rand() - h_bounds
        decide = pdf.max() * torch.rand()

        maxed = torch.exp(
            torch.sum(coeffs * torch.pow(loc, torch.arange(coeffs.shape[0])))
        )

        if decide < maxed:
            rnd_sample[i_sample] = loc
            i_sample += 1

    return rnd_sample
