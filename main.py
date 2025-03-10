#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

import plot_utils
import math_utils


def main():
    # constants
    GSTEPS = 1000
    E_A, E_D = 7, 5
    E_VENT = math_utils.arange(-3, 13, 1000)
    ALPHA = 1
    GSTART, GSTOP = -25, 25
    INIT_dG0 = 0

    # initial energies
    U_D = math_utils.get_potential_energy(ALPHA, E_D, E_VENT)
    U_A = math_utils.get_potential_energy(ALPHA, E_A, E_VENT)
    dG0_vals = math_utils.arange(GSTART, GSTOP, GSTEPS)
    min_idx = plot_utils.get_index(U_D, U_A - INIT_dG0)
    energies, rate, time_vals = plot_utils.get_lines(
        min_idx, dG0_vals, U_D, ALPHA, E_A, E_D, E_VENT
    )

    # set initial plots
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    point = ax1.scatter(
        *plot_utils.get_poi(min_idx, E_VENT, U_D), color="red", label=r"$\dagger$"
    )
    ax1.plot(E_VENT, U_D, color="black", label=r"$U_D$")
    (line,) = ax1.plot(E_VENT, U_A - INIT_dG0, label=r"$U_A$", color="purple")
    (energy,) = ax2.plot(
        dG0_vals,
        energies,
        label=r"$\Delta G^\dagger$",
    )
    (rates,) = ax2.plot(dG0_vals, rate, label=r"$K^\dagger$")
    (time,) = ax2.plot(dG0_vals, time_vals, label="time")
    vline = ax2.axvline(INIT_dG0, label=r"$\dagger$", ls="--", color="red")

    # axis settings
    ax2.set_xlim([-15, 15])
    ax2.set_ylim([-30, 30])
    ax1.set_ylim([-5, 35])
    ax1.legend()
    ax2.legend()

    # construct sliders
    dG0_slider, E_A_slider, alpha_slider = plot_utils.get_sliders(
        fig, GSTART, GSTOP, INIT_dG0, E_A, ALPHA
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        # update parameters from sliders
        delta_dG0 = dG0_slider.val
        E_A = E_A_slider.val
        alpha = alpha_slider.val

        # recompute with updated parameters
        U_A = math_utils.get_potential_energy(alpha, E_A, E_VENT)
        min_idx = plot_utils.get_index(U_D, U_A - delta_dG0)
        energies, rate, time_vals = plot_utils.get_lines(
            min_idx, dG0_vals, U_D, alpha, E_A, E_D, E_VENT
        )

        # update lines with new data
        line.set_ydata(U_A - delta_dG0)
        energy.set_ydata(energies)
        rates.set_ydata(rate)
        time.set_ydata(time_vals)
        vline.set_xdata(delta_dG0)
        fig.canvas.draw_idle()
        if not np.isnan(min_idx):  # check if intersection exists
            point.set_offsets(plot_utils.get_poi(min_idx, E_VENT, U_D))
        else:
            point.set_offsets([np.nan, np.nan])

    dG0_slider.on_changed(update)
    E_A_slider.on_changed(update)
    alpha_slider.on_changed(update)
    plt.show()
    return


main()
