import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from numba import njit, jit
from typing import Set, Tuple, Type
import itertools

from src.dataclass import (Input, Lattice, Parameter, Train, Save,
                           Processed_Input, Topology, Conjugate, Result)


def get_processed_input(input: Input) -> Processed_Input:
    topology = get_topology(input.lattice)

    return Processed_Input(topology)


def get_topology(lattice: Lattice) -> Topology:

    size, dimension = (
        lattice.size,
        lattice.dimension,
    )

    coordinate, interaction_point, distance, irreducible_distance = get_lattice_structure(
        size, dimension)

    return Topology(coordinate, interaction_point, distance, irreducible_distance)


@njit
def get_lattice_structure(size, dimension):
    coordinate = np.empty((size**dimension, dimension), dtype=np.int64)

    for i in range(size**dimension):
        for j in range(dimension):
            coordinate[i, j] = int((i % size**(j + 1)) / size**j)

    # add coordinate of point i
    if size == 2:
        interaction_point = np.empty(
            (size**dimension, dimension), dtype=np.int64)
        for i in range(size**dimension):
            for j in range(dimension):
                interaction_point[i, j] = i + (
                    (coordinate[i, j] + 1) % size - coordinate[i][j]) * size**j

    else:
        interaction_point = np.empty(
            (size**dimension, 2*dimension), dtype=np.int64)
        for i in range(size**dimension):
            for j in range(dimension):
                interaction_point[i, 2*j] = i + (
                    (coordinate[i, j] + 1) % size - coordinate[i][j]) * size**j
                interaction_point[i, 2*j+1] = i + (
                    (coordinate[i, j] + size - 1) % size - coordinate[i][j]) * size**j

    # add distance |i-j| between point i and j
    distance = np.empty((size**dimension, size**dimension), np.float64)
    for i in range(size**dimension):
        for j in range(i, size**dimension):
            distance1 = np.abs(coordinate[i] - coordinate[j])
            distance2 = size - distance1
            distance[j, i] = distance[i, j] = (np.minimum(
                distance1, distance2)).sum()
            # distance[i, j] = (np.minimum(distance1, distance2)**2).sum()

    # add irreducible list of distance |i-j| for point i and j
    irreducible_distance = [0]
    for i in range(size**dimension):
        if distance[0][i] not in irreducible_distance:
            irreducible_distance.append(distance[0][i])
    irreducible_distance = np.array(sorted(irreducible_distance))

    return coordinate, interaction_point, distance, irreducible_distance


def get_T_and_H(input: Input) -> tuple[float, float]:
    parameter = input.parameter

    Tc, Hc, T, H, mode, variable, multiply, base, exponent = (
        parameter.Tc,
        parameter.Hc,
        parameter.T,
        parameter.H,
        parameter.mode,
        parameter.variable,
        parameter.multiply,
        parameter.base,
        parameter.exponent,
    )

    if mode == "critical":
        return Tc, Hc

    elif mode == "normal":
        if variable == "T":
            if exponent >= 0:
                return Tc + multiply * base**exponent, Hc
            elif exponent < 0:
                return Tc - multiply * base ** (-exponent), Hc
        elif variable == "H":
            if exponent >= 0:
                return Tc, Hc + multiply * base**exponent
            elif exponent < 0:
                return Tc, Hc - multiply * base ** (-exponent)
        raise ValueError("variable should be 'T' or 'H'")

    elif mode == "manual":
        return T, H

    raise ValueError(
        "mode should be 'critical' or 'exponential' or 'linear' or 'manual'")


def get_J(input: Input, processed_input: Processed_Input) -> npt.NDArray:
    lattice, parameter, topology = (
        input.lattice, input.parameter, processed_input.topology
    )

    size, dimension, Jm, Jv, interaction_point = (
        lattice.size,
        lattice.dimension,
        parameter.Jm,
        parameter.Jv,
        topology.interaction_point,
    )

    rng = np.random.default_rng()

    J = np.zeros((size**dimension, size**dimension), dtype=np.float64)

    for i in range(size**dimension):
        for j in interaction_point[i]:
            if j >= i:
                J[i, j] = rng.normal(Jm, Jv, 1)
                J[j, i] = J[i, j].copy()

    return J
