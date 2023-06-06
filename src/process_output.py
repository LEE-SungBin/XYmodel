from src.function import (
    magnetization,
    get_spin_glass,
    hamiltonian,
    kurtosis,
    time_correlation,
    space_correlation,
)
from src.dataclass import (
    Input, Lattice, Parameter, Train, Save,
    Processed_Input, Topology, Conjugate, Result
)


import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
import time


def get_result(
    input: Input,
    processed_input: Processed_Input,
    raw_output: npt.NDArray[np.complex128],
    J: npt.NDArray,
) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, npt.NDArray]:

    now = time.perf_counter()
    order, suscept, binder, spin_order, spin_suscept, spin_binder = get_order_parameter(
        input, processed_input, raw_output)
    # print(f"order parameter process completed, {time.perf_counter()-now}s")

    now = time.perf_counter()
    energy, specific = get_total_energy(input, processed_input, raw_output, J)
    # print(f"energy process completed, {time.perf_counter()-now}s")

    now = time.perf_counter()
    correlation = get_correlation_function(input, processed_input, raw_output)
    # print(
    # f"correlation function process completed, {time.perf_counter()-now}s")

    return order, suscept, binder, spin_order, spin_suscept, spin_binder, energy, specific, correlation


def get_animation(
    input: Input,
    raw_output: npt.NDArray[np.complex128],
) -> None:
    fps, size, dimension, measurement, T = (
        30,
        input.lattice.size,
        input.lattice.dimension,
        input.train.measurement,
        input.parameter.T,
    )

    if (dimension != 2):
        raise ValueError("animation only for two dimension")

    output = np.angle(raw_output).reshape(measurement, size, size)

    def get_arrow(i):
        u, v, M = np.cos(output[i]), np.sin(output[i]), output[i]
        return u, v, M

    fig, ax = plt.subplots(figsize=(10, 10))

    x, y = np.mgrid[0: size, 0: size]
    cmap = plt.cm.get_cmap('hsv')
    quiver = ax.quiver(
        x, y, *get_arrow(0), pivot='mid',
        width=0.0025*64/size, scale_units='x', scale=0.6,
        norm=colors.Normalize(vmin=-np.pi, vmax=np.pi), cmap=cmap
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(quiver, cax=cax)
    # ax.scatter(x,y, c='red', s=2.7)

    def update(i):  # return time is tuple, i.e., im, or [im]
        print(i, end=" ")
        # global quiver
        # quiver.remove()
        ax.cla()  # ! clear current axis
        ax.quiver(
            x, y, *get_arrow(i), pivot='mid',
            width=0.0025*64/size, scale_units='x', scale=0.6,
            norm=colors.Normalize(vmin=-np.pi, vmax=np.pi), cmap=cmap
        )
        ax.text(
            0.99, 0.01, f"{i+1}",
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=20
        )
        # plt.colorbar(quiver, cax=cax)
        ax.set_title(
            rf"XY | Size = {size} x {size} | T = {np.round(T,3)} | fps = {fps}", fontsize=25)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)

    anim = animation.FuncAnimation(
        fig, update, frames=measurement, interval=1000/fps, blit=False
    )  # interval in ms

    # anim.save(f"./ani/XY, {size} x {size}, {np.round(T,3)}.gif", fps=fps)

    anim.save(f"./ani/XY, {size} x {size}, {np.round(T,3)}.mp4", fps=fps,
              extra_args=['-vcodec', 'libx264'])

    # plt.show()


def get_grad_animation(
    input: Input,
    processed_input: Processed_Input,
    raw_output: npt.NDArray[np.complex128],
) -> None:
    fps, size, dimension, measurement, T, nn_coord = (
        30,
        input.lattice.size,
        input.lattice.dimension,
        input.train.measurement,
        input.parameter.T,
        processed_input.topology.interaction_point,
    )

    if (dimension != 2):
        raise ValueError("animation only for two dimension")

    output = np.angle(raw_output).reshape(measurement, -1)

    def get_arrow(i):
        u, v, M = np.cos(output[i]), np.sin(output[i]), output[i]
        return u, v, M

    fig, ax = plt.subplots(figsize=(10, 10))

    x, y = np.mgrid[0: size, 0: size]
    cmap = plt.cm.get_cmap('hsv')
    quiver = ax.quiver(
        x, y, *get_arrow(0), pivot='mid',
        width=0.0025*64/size, scale_units='x', scale=0.6,
        norm=colors.Normalize(vmin=-np.pi, vmax=np.pi), cmap=cmap
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(quiver, cax=cax)
    # ax.scatter(x,y, c='red', s=2.7)

    def update(i):  # return time is tuple, i.e., im, or [im]
        print(i, end=" ")
        # global quiver
        # quiver.remove()
        ax.cla()  # ! clear current axis
        ax.quiver(
            x, y, *get_arrow(i), pivot='mid',
            width=0.0025*64/size, scale_units='x', scale=0.6,
            norm=colors.Normalize(vmin=-np.pi, vmax=np.pi), cmap=cmap
        )
        ax.text(
            0.99, 0.01, f"{i+1}",
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=20
        )
        # plt.colorbar(quiver, cax=cax)
        ax.set_title(
            rf"XY | Size = {size} x {size} | T = {np.round(T,3)} | fps = {fps}", fontsize=25)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)

    anim = animation.FuncAnimation(
        fig, update, frames=measurement, interval=1000/fps, blit=False
    )  # interval in ms

    # anim.save(f"./ani/XY, {size} x {size}, {np.round(T,3)}.gif", fps=fps)

    anim.save(f"./ani/XY, {size} x {size}, {np.round(T,3)}.mp4", fps=fps,
              extra_args=['-vcodec', 'libx264'])

    # plt.show()


@njit
def get_gradient(
    angles,
    nn_coord
):
    x = 1


def get_order_parameter(
        input: Input,
        processed_input: Processed_Input,
        raw_output: npt.NDArray[np.complex128],
) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]:

    size, dimension, ghost, T = (
        input.lattice.size,
        input.lattice.dimension,
        input.lattice.ghost,
        input.parameter.T,
    )
    order = magnetization(raw_output, ghost)
    spin_glass = get_spin_glass(raw_output)

    return (
        np.sqrt(np.average(order).real**2+np.average(order).imag**2),
        np.std(order)**2 * size**dimension / T,
        1 - kurtosis(order) / 3.0,
        np.average(spin_glass),
        np.std(spin_glass)**2 * size**dimension / T,
        1 - kurtosis(spin_glass) / 3.0,
    )


def get_total_energy(
        input: Input,
        processed_input: Processed_Input,
        raw_output: npt.NDArray[np.complex128],
        J: npt.NDArray,
) -> tuple[np.float64, np.float64]:

    size, dimension, ghost, T, H = (
        input.lattice.size,
        input.lattice.dimension,
        input.lattice.ghost,
        input.parameter.T,
        input.parameter.H,
    )

    temp = hamiltonian(
        raw_output, ghost, J, H)

    return np.average(temp), np.std(temp) ** 2 * size**dimension / T**2


def get_correlation_function(
    input: Input,
    processed_input: Processed_Input,
    raw_output: npt.NDArray[np.complex128],
) -> npt.NDArray:

    distance, irreducible_distance = (
        processed_input.topology.distance,
        processed_input.topology.irreducible_distance,
    )

    now = time.perf_counter()
    G_ij = space_correlation(raw_output)  # G(i,j)
    # print(
    #     f"G(i,j) process completed, {time.perf_counter()-now}s, {np.size(G_ij)}")

    now = time.perf_counter()

    # correlation = process_correlation_function(
    #     G_ij, irreducible_distance, distance)

    correlation = np.zeros_like(irreducible_distance)
    for i, irr in enumerate(irreducible_distance):
        correlation[i] = G_ij[(distance == irr)].mean()

    # print(f"correlation = {correlation}")
    # print(f"G(|i-j|) process completed, {time.perf_counter()-now}s")

    return correlation


# @njit
# def process_correlation_function(G_ij, irreducible_distance, distance) -> npt.NDArray:
#     result = np.zeros_like(irreducible_distance)
#     for i, irr in enumerate(irreducible_distance):
#         row, col = np.where(distance == irr)
#         for r, c in zip(row, col):
#             result[i] += G_ij[r, c]
#         result[i] /= len(row)
#
#     return result
