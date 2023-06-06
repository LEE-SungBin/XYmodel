import numpy as np
import numpy.typing as npt
from numba import jit, njit, prange


def kurtosis(
    arr: npt.NDArray
) -> np.float64:
    length = len(arr)

    return np.real(
        np.einsum("i,i,i,i->", np.conjugate(arr), arr,
                  np.conjugate(arr), arr, optimize=True) / length)/np.real(
        np.einsum("i,i->", np.conjugate(arr), arr, optimize=True)/length)**2


def magnetization(
    array: npt.NDArray[np.complex128],
    ghost: float,
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    conjugate_ghost: [size**dim]
    """

    length = np.size(array[0])
    conjugate_ghost = np.full(np.size(array[0]), np.exp(-ghost*1j))

    # return np.real(
    #     np.tensordot(conjugate_ghost, array, (0, 1)) / length
    # )

    return np.einsum("ij->i", array, optimize=True)/length


def get_spin_glass(
    array: npt.NDArray[np.complex128],
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    """

    measurement = np.size(array[:, 0])

    spin_glass = np.einsum(
        "ij->j", array, optimize=True) / measurement

    return np.real(spin_glass*np.conjugate(spin_glass))


def hamiltonian(
    array: npt.NDArray[np.complex128],
    ghost: float,
    J: npt.NDArray,
    H: float,
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    conjugate_ghost: [size**dim]
    J: [size**dim, size**dim]
    """

    conjugate_ghost = np.full(np.size(array[0]), np.exp(-ghost*1j))

    return np.real(
        - H * np.tensordot(conjugate_ghost, array, (0, 1))
        - np.einsum("ji,ij->i", np.tensordot(J, array, (0, 1)),
                    np.conjugate(array), optimize=True) / 2.0
    ) / np.size(array[0])


# return autocorrelation <sigma(t=0),sigmma(t)>
def time_correlation(
    arr1: npt.NDArray[np.complex128],
    arr2: npt.NDArray[np.complex128],
    length: int
) -> float:
    return np.real(np.vdot(arr1, arr2)).item() / length


# return connected-correlation <sigma(i),sigma(j)>-<sigma(i)><sigma(j)> between two point in arr
def space_correlation(
    array: npt.NDArray[np.complex128],
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    """
    measurement = np.size(array[:, 0])

    average = np.einsum("ij->j", array, optimize=True) / measurement
    corr = np.tensordot(np.conjugate(array), array, (0, 0)) / measurement

    return np.real(corr - np.tensordot(np.conjugate(average), average, axes=0))


def column_average_2d(arrs: list[np.ndarray]) -> npt.NDArray[np.float64]:
    # return np.array([np.abs(arr.mean()) for arr in arrs])
    # return arr.mean(axis=1)

    length = []
    for row in arrs:
        length.append(np.size(row))

    size = np.max(np.array(length))
    temp = [[] for _ in range(size)]

    for row in arrs:
        for i in range(np.size(row)):
            temp[i].append(row[i])

    return np.array([abs(np.average(temp[i])) for i in range(size)])
