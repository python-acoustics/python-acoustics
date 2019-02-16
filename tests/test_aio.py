import os
import itertools

import pytest
import numpy as np

from get_data_path import data_path

details_type = ["det_global", "det_oct", "det_third"]
int_time = ["2000ms", "1000ms", "0500ms", "0250ms", "0063ms", "0010ms"]
delimiter = ["comma", "semicolon"]
decimals = ["1dec", "2dec"]
time_history = ["overall", "timehist"]
weighting = ["z", "za", "zc", "zac"]
cirrus_files = list(itertools.product(
    details_type,
    int_time,
    delimiter,
    decimals,
))
cirrus_files.extend(list(itertools.product(
    ["det_global_general"],
    delimiter,
    decimals,
)))
cirrus_files.extend(list(itertools.product(
    time_history,
    ["oct"],
    weighting,
    delimiter,
    decimals,
)))
cirrus_files.extend(list(itertools.product(
    ["overall_third"],
    weighting,
    delimiter,
    decimals,
)))


@pytest.mark.parametrize('filename', cirrus_files)
def test_read_csv_cirrus_details(filename):
    from acoustics.aio import read_csv_cirrus
    file = "_".join(filename) + ".csv"
    csv_path = os.path.join(data_path(), "cirrus", file)
    data = read_csv_cirrus(csv_path)
    if filename[0] == "det_global":
        np.sum(data.LAeq)
    else:
        np.sum(data["125Hz"])
