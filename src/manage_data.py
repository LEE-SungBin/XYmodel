import numpy as np
import numpy.typing as npt
import string
from dataclasses import dataclass, asdict, field
import hashlib
import json
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
from typing import Any, Optional, Union
import pickle
from datetime import datetime
import os

from src.dataclass import (Input, Lattice, Parameter, Train, Save,
                           Processed_Input, Topology, Conjugate, Result)

from pathlib import Path


@dataclass
class Data:
    size: int
    dimension: int
    Jv: float
    temperature: npt.NDArray
    order_parameter: npt.NDArray
    susceptibility: npt.NDArray
    binder_cumulant: npt.NDArray
    spin_glass_order: npt.NDArray
    spin_glass_suscept: npt.NDArray
    spin_glass_binder: npt.NDArray
    energy: npt.NDArray
    specific_heat: npt.NDArray
    irreducible_distance: npt.NDArray
    correlation_function: npt.NDArray
    correlation_length: npt.NDArray


def save_result(input: Input, result: Result) -> None:
    #! Hash vs random key
    # From input, create directory to store input/result
    key = hashlib.sha1(str(input).encode()).hexdigest()[:6]
    # key = "".join(
    #     np.random.choice(list(string.ascii_lowercase + string.digits), 6)
    # )

    dir_path = Path(f"./{input.save.location}")
    dir_path.mkdir(parents=True, exist_ok=True)

    output = {
        "key": key,
    }

    output.update(asdict(input.lattice))
    output.update(asdict(input.parameter))
    output.update(asdict(input.train))
    # output.update(asdict(input.correlation))
    output.update(asdict(input.save))
    output.update(asdict(result))

    with open(dir_path / f"{key}.pkl", "wb") as file:
        pickle.dump(output, file)


def save_log(input: Input, result: Result) -> None:

    log = (
        f"{datetime.now().replace(microsecond=0)} {input.to_log()} {result.to_log()}\n"
    )

    dir_path = Path(".")
    dir_path.mkdir(parents=True, exist_ok=True)

    with open(dir_path / "log.txt", "a") as file:
        file.write(log)


def load_result(location: str) -> pd.DataFrame:
    dir_path = Path(f"./{location}")
    filenames = [f for f in listdir(dir_path) if isfile(
        join(dir_path, f)) if ".pkl" in f]
    list_result = []
    for filename in filenames:
        result = join(dir_path, filename)
        if os.path.getsize(result) > 0:
            with open(result, "rb") as file:
                results = pickle.load(file)
                list_result.append(results)

    df = pd.DataFrame(list_result)  # .drop(columns=[])
    return df


# tuple[pd.DataFrame, Data]
def get_result(df: pd.DataFrame, dimension: int, size: int, Jv: float) -> Data:
    filter_ = (df["size"] == size) & (
        df["dimension"] == dimension) & (df["Jv"] == Jv)

    frame = df[filter_].sort_values("T", ascending=True)

    return Data(
        size=size,
        dimension=dimension,
        Jv=Jv,
        temperature=np.array(frame["T"]),
        order_parameter=np.array(frame["order_parameter"]),
        susceptibility=np.array(frame["susceptibility"]),
        binder_cumulant=np.array(frame["binder_cumulant"]),
        spin_glass_order=np.array(frame["spin_glass_order"]),
        spin_glass_suscept=np.array(frame["spin_glass_suscept"]),
        spin_glass_binder=np.array(frame["spin_glass_binder"]),
        energy=np.array(frame["energy"]),
        specific_heat=np.array(frame["specific_heat"]),
        irreducible_distance=np.array(frame["irreducible_distance"]),
        correlation_function=np.array(frame["correlation_function"]),
        correlation_length=np.zeros(np.size(np.array(frame["T"]))),
    )


def get_correlation_length(data: Data, x_min=None, x_max=None, y_min=None) -> npt.NDArray:
    irreducible_distance, correlation_function = (
        data.irreducible_distance,
        data.correlation_function,
    )

    if x_min is None:
        x_min = 0.0
    if x_max is None:
        x_max = data.size/2/np.sqrt(2)
    if y_min is None:
        y_min = 1.e-7

    correlation_length = []

    for i, distance_list in enumerate(irreducible_distance):
        correlation = correlation_function[i]
        x, y = [], []
        for j, distance in enumerate(distance_list):
            if (x_min <= distance <= x_max and correlation[j] >= y_min):
                x.append(distance)
                y.append(np.log(correlation[j]))

        x, y = np.array(x).reshape((-1, 1)), np.array(y)

        # print(x, y)
        model = LinearRegression().fit(x, y)
        correlation_length.append(-1.0/model.coef_)

    correlation_length = np.array(correlation_length)
    return correlation_length


def get_label(data: Data, mode: str) -> str:
    if (mode == "size"):
        return get_size(data)
    elif (mode == "Jv"):
        return get_Jv(data)
    raise ValueError("label mode should be either 'size' or 'Jv'")


def get_list(df: pd.DataFrame, list_name: str) -> list:
    data_list = []

    for data in df[f"{list_name}"]:
        if (data not in data_list):
            data_list.append(data)

    data_list.sort()

    return data_list


def get_size(data: Data) -> str:
    size_print = f"Size = {data.size} "

    for _ in range(data.dimension-1):
        size_print += f"x {data.size}"

    return size_print


def get_Jv(data: Data) -> str:

    return f"Jv = {data.Jv}"


def delete_result(location: str, key_name: npt.NDArray) -> None:
    import os

    # Try to delete the file.

    for key in key_name:
        try:
            os.remove(f"./{location}/{key}.pkl")
        except OSError as e:
            # If it fails, inform the user.
            print("Error: %s - %s." % (e.filename, e.strerror))
