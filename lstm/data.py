"""Data loading and preprocessing helpers for the LSTM experiments."""
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path("00_DatosOriginalesLag.xlsx")


def open_file(file_path: Path | str = DEFAULT_DATA_PATH,
              split_point: Tuple[int, int] = (3230, 4010),
              beta: str = "1") -> np.ndarray:
    """
    Load the nominal beta series from the Excel workbook and return as a 1D array.

    Parameters
    ----------
    file_path: Path or str
        Location of the Excel workbook containing the `Nominales_B{beta}_Y` sheet.
    split_point: tuple
        Inclusive start and exclusive end indices to slice from the source sheet.
    beta: str
        Beta identifier used to construct the sheet name.
    """
    dataset = pd.read_excel(file_path, header=0, sheet_name=f"Nominales_B{beta}_Y")
    dataset = dataset.iloc[split_point[0]:split_point[1]]
    dataset = dataset.values.astype("float32")

    sequence: list[float] = []
    for row in dataset:
        sequence.append(row[0])

    return np.array(sequence)


# Backwards-compatible alias kept from the original notebook.
def openFile(file_path: Path | str = DEFAULT_DATA_PATH,
             splitPoint: Tuple[int, int] = (3230, 4010),
             Beta: str = "1") -> np.ndarray:
    return open_file(file_path=file_path, split_point=splitPoint, beta=Beta)


def split_sequence(sequence: Sequence[float], n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a univariate sequence into samples of length ``n_steps`` and their targets."""
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
