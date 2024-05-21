#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider

import numpy as np


def get_potential_energy(alpha, E_nu, E: np.ndarray):
    return ((E - E_nu) ** 2) / (2 * alpha)


def get_lambda(alpha, E_A, E_D, delta_G0, E_vent: np.ndarray):
    U_A = get_potential_energy(alpha, E_A, E_vent) - delta_G0
    U_D = get_potential_energy(alpha, E_D, E_vent)
    return ((U_A - U_D) ** 2) / (2 * alpha)


def get_delta_G_ddagger(delta_G0, lam):
    return ((delta_G0 - lam) ** 2) / (4 * lam)


def get_Ket(delta_G_ddagger, T=293):
    kB = 1.380649e-23
    return (
        -delta_G_ddagger
    )  # -1 / delta_G_ddagger  # -1 / delta_G_ddagger  # / (kB * T * 1000000000))


def main():
    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes[0], axes[1]

    E_A, E_D = 7, 5
    E_vent = np.arange(-3, 13, 0.05)
    alpha = 1
    # energies = []
    rates = []
    U_D = get_potential_energy(alpha, E_D, E_vent)
    U_A = get_potential_energy(alpha, E_A, E_vent)

    init_dG0 = 0
    ax1.plot(E_vent, U_D, color="black", label=r"$U_D$")
    (line,) = ax1.plot(E_vent, U_A - init_dG0, label=r"$U_A$", color="purple")
    # (k,) = ax2.plot()

    def get_index(alpha, E_A, E_D, delta_G0, E_vent):
        lam = get_lambda(alpha, E_A, E_D, delta_G0, E_vent)
        min_idx = np.argmin(lam)
        return min_idx

    def get_poi(min_index, E_vent, U_D):
        return E_vent[min_index], U_D[min_index]

    min_index = get_index(alpha, E_A, E_D, init_dG0, E_vent)
    point = ax1.scatter(
        *get_poi(min_index, E_vent, U_D), color="red", label=r"$\dagger$"
    )

    start, stop, step = -25, 25, 0.1
    dG0_vals = np.arange(start, stop + step, step)
    ax_dG0 = fig.add_axes([0.25, 0.0025, 0.63, 0.0225])
    dG0_slider = Slider(
        ax=ax_dG0,
        label=r"$\Delta G_o$",
        valmin=start,
        valmax=stop,
        valinit=init_dG0,
        orientation="horizontal",
    )
    ax_E_A = fig.add_axes([0.25, 0.04, 0.63, 0.0225])
    E_A_slider = Slider(
        ax=ax_E_A,
        label=r"$E_A$",
        valmin=E_A - 10,
        valmax=E_A + 10,
        valinit=E_A,
        orientation="horizontal",
    )
    ax_alpha = fig.add_axes([0.025, 0.25, 0.0225, 0.63])
    alpha_slider = Slider(
        ax=ax_alpha,
        label=r"$\alpha$",
        valmin=0,
        valmax=2,
        valinit=1,
        orientation="vertical",
    )
    ax2.set_xlim([-20, 20])

    def get_delta_G_ddagger(dG0_vals, alpha, E_A, E_D, delta_dG0, E_vent) -> np.ndarray:
        energies = []
        for delta_dG0 in dG0_vals:
            min_idx = get_index(alpha, E_A, E_D, delta_dG0, E_vent)
            delta_G_ddagger = U_D[min_idx] - min(
                U_D
            )  # get_delta_G_ddagger(delta_G0, lam[min_idx])
            energies.append(delta_G_ddagger)
        return np.array(energies)

    def get_rate(dG0_vals, alpha, E_A, E_D, delta_dG0, E_vent) -> np.ndarray:
        energies = []
        for delta_dG0 in dG0_vals:
            min_idx = get_index(alpha, E_A, E_D, delta_dG0, E_vent)
            delta_G_ddagger = U_D[min_idx] - min(
                U_D
            )  # get_delta_G_ddagger(delta_G0, lam[min_idx])
            energies.append(-delta_G_ddagger)
        return np.array(energies)

    energies = get_delta_G_ddagger(dG0_vals, alpha, E_A, E_D, init_dG0, E_vent)
    rate = get_rate(dG0_vals, alpha, E_A, E_D, init_dG0, E_vent)

    idx = np.isclose(
        energies, np.max(energies), atol=1
    )  # NOTE: filter out values out of the range we calculate
    energies[idx] = np.nan
    dG0_vals[idx] = np.nan
    rate[idx] = np.nan
    rate[rate > -0.025] = np.nan
    rate[rate > -0.025] = np.nan

    (energy,) = ax2.plot(
        dG0_vals,
        get_delta_G_ddagger(dG0_vals, alpha, E_A, E_D, init_dG0, E_vent),
        label=r"$\Delta G^\dagger$",
    )
    (rates,) = ax2.plot(dG0_vals, rate, label=r"$K^\dagger$")
    (time,) = ax2.plot(dG0_vals, 1 / rate, label="time")
    vline = ax2.axvline(init_dG0, label=r"$\dagger$", ls="--", color="red")

    ax1.legend()
    ax2.legend()

    # The function to be called anytime a slider's value changes
    def update(val):
        dG0_vals = np.arange(start, stop + step, step)

        # update parameters from sliders
        delta_dG0 = dG0_slider.val
        E_A = E_A_slider.val
        alpha = alpha_slider.val

        # recompute with updated parameters
        U_A = get_potential_energy(alpha, E_A, E_vent)
        min_index = get_index(alpha, E_A, E_D, delta_dG0, E_vent)
        energies = get_delta_G_ddagger(dG0_vals, alpha, E_A, E_D, delta_dG0, E_vent)
        rate = get_rate(dG0_vals, alpha, E_A, E_D, delta_dG0, E_vent)

        idx = np.isclose(
            energies, np.max(energies), atol=1
        )  # NOTE: filter out values out of the range we calculate
        energies[idx] = np.nan
        dG0_vals[idx] = np.nan
        rate[idx] = np.nan
        rate[rate > -0.025] = np.nan

        # update lines with new data
        line.set_ydata(U_A - delta_dG0)
        point.set_offsets(get_poi(min_index, E_vent, U_D))
        energy.set_ydata(energies)
        rates.set_ydata(rate)
        time.set_ydata(1 / rate)
        vline.set_xdata(delta_dG0)
        fig.canvas.draw_idle()

    dG0_slider.on_changed(update)
    E_A_slider.on_changed(update)
    alpha_slider.on_changed(update)
    plt.show()
    return


main()
