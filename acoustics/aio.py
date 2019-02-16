"""
Cirrus
======

Handle Cirrus data.

"""
import csv
import io
import re

import pandas as pd


def read_csv_cirrus(filename):  # pylint: disable=too-many-locals
    """Read a Cirrus CSV file. Currently exists support for some types of
    CSV files extracted with NoiseTools. There is no support for CSVs related
    with occupational noise.

    If there are NC and NR values in the csv file, they will be stored in the
    returned object with attributes ``nc`` and ``nr``. If the CSV file contains
    time history, you can access to date and time with the ``time`` attribute.
    Also, it is possible to know the integration time with the
    ``integration_time`` attribute.

    :param filename: CSV file name.
    :returns: Pandas dataframe with all data extracted from the CSV file.
    :rtype: Pandas dataframe.

    """
    with open(filename, "r") as csvfile:
        csvreader = csvfile.read()
        csvreader = re.sub(r" dB", "", csvreader)  # Clean " dB" from data
        dialect = csv.Sniffer().sniff(csvreader, delimiters=",;")
        separator = dialect.delimiter
        # Guess decimal separator
        decimal_sep = re.search(
            r"\"\d{2,3}"
            r"(\.|,)"  # Decimal separator
            r"\d{1,2}\"",
            csvreader,
        ).group(1)
    n_cols = re.search("(.+)\n", csvreader).group(1).count(separator) + 1
    if n_cols < 5:
        unsorted_data = []
        pdindex = ["Z"]
        for i, c in enumerate(csvreader.splitlines()):
            if c[:4] == '"NR"':
                nr = int(re.search(r"\d{2}", c).group(0))
                continue
            elif c[:4] == '"NC"':
                nc = int(re.search(r"\d{2}", c).group(0))
                continue
            if i != 0:
                unsorted_data.append(c.split(separator))
            else:
                if n_cols == 3:
                    pdindex.append(c[-2:-1])
                elif n_cols == 4:
                    pdindex.append("A")
                    pdindex.append("C")

        # Create a sorted temporary csv-like file
        csv_data = list(zip(*unsorted_data))
        temp_csv = ""
        for row in csv_data:
            temp_csv += separator.join(row) + "\n"
        # Then, read it with pandas
        data = pd.read_csv(
            io.StringIO(temp_csv),
            sep=separator,
            decimal=decimal_sep,
        )

        # Assign NC and NR data if they are present
        try:
            data.nc = nc
            data.nr = nr
        # TODO specify exception type:
        except:  # pylint: disable=bare-except
            pass

        # If the csv file contains global data from the "Details" tab in
        # NoiseTools, skip row names
        if n_cols != 2:
            data.index = pdindex

    else:
        data = pd.read_csv(
            filename,
            parse_dates=[[0, 1]],
            sep=separator,
            decimal=decimal_sep,
        )

        # Fix time name column
        en_columns = data.columns.values
        en_columns[0] = "time"
        data.columns = en_columns

        # Guess integration time with statistical mode because the csv could
        # have been cleaned from unwanted noise
        data["time"] = pd.to_datetime(data.time)
        delta = data.time.diff().fillna(0.0)
        # Mode and change from ns to s
        int_time = int(delta.mode().astype(int) * 1e-9)
        if round(int_time, 2) == 0.06:  # Fix for 1/16 s
            int_time = 0.0625
        data.integration_time = int_time

    return data
