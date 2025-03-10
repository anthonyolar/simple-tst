#!/usr/bin/env python3
import numpy as np


def get_potential_energy(alpha: float, E_nu: float, E: np.ndarray) -> np.ndarray:
    return ((E - E_nu) ** 2) / (2 * alpha)


def get_lambda(
    alpha: float, E_A: float, E_D: float, delta_G0: float, E_VENT: np.ndarray
) -> np.ndarray:
    U_A = get_potential_energy(alpha, E_A, E_VENT) - delta_G0
    U_D = get_potential_energy(alpha, E_D, E_VENT)
    return ((U_A - U_D) ** 2) / (2 * alpha)


def get_delta_G_ddagger(
    dG0_vals: np.ndarray,
    U_D: np.ndarray,
    alpha: float,
    E_A: float,
    E_D: float,
    E_VENT: np.ndarray,
) -> np.ndarray:
    energies = []
    for delta_dG0 in dG0_vals:
        min_idx = np.argmin(get_lambda(alpha, E_A, E_D, delta_dG0, E_VENT))
        delta_G_ddagger = U_D[min_idx] - min(U_D)
        energies.append(delta_G_ddagger)
    return np.array(energies)


def arange(start: float, stop: float, steps: int = 100) -> np.ndarray:
    stepsize = (stop - start) / steps
    return np.arange(start, stop + stepsize, stepsize)
