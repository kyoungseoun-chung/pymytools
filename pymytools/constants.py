#!/usr/bin/env python3
"""
Store all relevant constants. It contains

* Electron mass
* Proton mass
* Neutron mass
* Atomic mass
* Speed of light
* Magnetic constant
* Vacuum permittivity
* Boltzmann constant
* Converting factor from eV to K
* Elementary charge
* Converting factor from K to eV
* Pi

"""
import numpy as _np
from scipy import constants as _constants

ME: float = _constants.m_e
"""Electron mass [kg]"""
MP: float = _constants.m_p
"""Proton mass [kg]"""
MN: float = _constants.m_n
"""Neutron mass [kg]"""
AMU: float = _constants.m_u
"""Atomic mass [kg]"""
C: float = _constants.c
"""Speed of light in vacuum [m/s]"""
MU0: float = _constants.mu_0
"""Magnetic constant [H/m]"""
EPS0: float = _constants.epsilon_0
"""Vacuum permittivity [F/m]"""
KB: float = _constants.k
"""Boltzmann constant [J/K]"""
eVtoK: float = _constants.value("electron volt-kelvin relationship")
"""Converting factor from eV to K"""
QE: float = _constants.value("elementary charge")
"""Elementary charge 1.60217662 × 10-19 [C]"""
KtoeV: float = 1.0 / eVtoK
"""Converting factor from K to eV"""
PI: float = _np.pi
"""PI"""
