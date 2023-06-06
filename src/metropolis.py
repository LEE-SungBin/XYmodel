from typing import Any
import numpy as np
import numpy.typing as npt
from numba import jit, njit, prange
import sys

from src.dataclass import (
    Input, Lattice, Parameter, Train, Save,
    Processed_Input, Topology, Conjugate, Result
)


def execute_metropolis_update(
    input: Input, processed_input: Processed_Input, J: npt.NDArray[np.float64], system_state: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    lattice, parameter, topology = (
        input.lattice,
        input.parameter,
        processed_input.topology,
    )

    size, dimension, T, H, interaction_point, complex_ghost = (
        lattice.size,
        lattice.dimension,
        parameter.T,
        parameter.H,
        topology.interaction_point,
        np.exp(lattice.ghost*1j),
    )

    Delta = np.pi

    rng = np.random.default_rng()
    flip_coord = np.arange(size**dimension)
    rng.shuffle(flip_coord)
    prob = rng.random(size=size**dimension)
    proposal = np.exp((2 * rng.random(size**dimension) - 1)
                      * Delta * 1j, dtype=np.complex128)

    return update_system_state(flip_coord, system_state, prob, proposal, H, T, complex_ghost, J, interaction_point)


@njit
def update_system_state(flip_coord, system_state, prob, proposal, H, T, complex_ghost, J, interaction_point):

    for i, x in enumerate(flip_coord):
        interaction = H * complex_ghost
        for point in interaction_point[x]:
            interaction += J[x][point] * system_state[point]

        update = system_state[x] * proposal[i]

        current_energy = np.real(- np.conjugate(system_state[x]) * interaction)
        flip_energy = np.real(- np.conjugate(update) * interaction)

        if flip_energy < current_energy:
            system_state[x] = update

        elif (prob[i] <= np.exp(- (flip_energy - current_energy) / T)):
            system_state[x] = update

    return system_state
