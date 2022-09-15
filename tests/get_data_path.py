import os
import pathlib


def data_path() -> pathlib.Path:
    parent = pathlib.Path(__file__).parent
    return parent / "data"
