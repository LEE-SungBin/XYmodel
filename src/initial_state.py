import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from numba import njit, jit
from typing import Set, Tuple, Type

from src.dataclass import (
    Input, Lattice, Parameter, Train, Save,
    Processed_Input, Topology, Conjugate, Result
)

# @njit(parallel=True)


def get_initial_state(
    input: Input
) -> npt.NDArray[np.complex128]:

    lattice = input.lattice

    if (lattice.initial == "uniform"):
        return np.full(lattice.size**lattice.dimension, np.exp(lattice.ghost*1j), dtype=np.complex128)

    elif (lattice.initial == "random"):
        rng = np.random.default_rng()

        return np.exp(rng.random(lattice.size**lattice.dimension)
                      * 2.0*np.pi*1j, dtype=np.complex128)

    raise ValueError("lattice.initial should be 'uniform' or 'random'")
