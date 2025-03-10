#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import math_utils


def smooth(x: np.ndarray) -> np.ndarray:
    smoothed = np.convolve(x, np.ones(20) / 20)
    return smoothed[9 : smoothed.size - 10]


def get_lines(
    min_idx: int,
    dG0_vals: np.ndarray,
    U_D: np.ndarray,
    alpha: float,
    E_A: float,
    E_D: float,
    E_VENT: np.ndarray,
):
    if np.isnan(min_idx):
        energies = np.zeros_like(dG0_vals)
        rate = np.zeros_like(dG0_vals)
        time_vals = np.zeros_like(dG0_vals)
    else:
        energies = math_utils.get_delta_G_ddagger(
            dG0_vals, U_D, alpha, E_A, E_D, E_VENT
        )
        filtered = energies.copy()
        filtered[filtered == 0] = 1e-8
        time_vals = -1 / filtered
        # smooth out the data
        energies = smooth(energies)
        rate = -energies
        time_vals = smooth(time_vals)
    return energies, rate, time_vals


def get_sliders(
    fig: plt.Figure, start: int, stop: int, INIT_dG0: float, E_A: float, alpha: float
):
    ax_dG0 = fig.add_axes([0.2, 0.0025, 0.63, 0.0225])
    dG0_slider = Slider(
        ax=ax_dG0,
        label=r"$\Delta G_o$",
        valmin=start,
        valmax=stop,
        valinit=INIT_dG0,
        orientation="horizontal",
    )
    ax_E_A = fig.add_axes([0.2, 0.04, 0.63, 0.0225])
    E_A_slider = Slider(
        ax=ax_E_A,
        label=r"$E_A$",
        valmin=E_A - 10,
        valmax=E_A + 10,
        valinit=E_A,
        orientation="horizontal",
    )
    ax_alpha = fig.add_axes([0.025, 0.175, 0.0225, 0.63])
    alpha_slider = Slider(
        ax=ax_alpha,
        label=r"$\alpha$",
        valmin=1e-1,
        valmax=2,
        valinit=alpha,
        orientation="vertical",
    )
    return dG0_slider, E_A_slider, alpha_slider


def get_index(curve_1: np.ndarray, curve_2: np.ndarray):
    intersections = np.argwhere(np.diff(np.sign(curve_1 - curve_2))).flatten()
    if intersections.size:
        return intersections[-1]
    return np.nan


def get_poi(min_idx: int, E_VENT: np.ndarray, U_D: np.ndarray):
    return E_VENT[min_idx], U_D[min_idx]
