import csv
import numpy as np
import pandas as pd
import datetime
import time


class FileDef(object):
    """An object class to pass as args to `load_csv_data`"""

    filename = None
    colname = None

    def __init__(self, filename, colname="value"):
        self.filename = filename
        self.colname = colname


def load_csv_data(
    *file_defs,
    date_colname="\ufeffDate",
    main_colname="PX_LAST",
    load_using_pandas=False
) -> pd.DataFrame:
    """Load market data from csv files as given in BBG files.
    Args:
        file_defs (FileDef) - FileDef objects with the file info
    Kwargs:
        date_colname (str, default: "\ufeffDate") - Name of the column
            containing date
        main_colname (str, default: "PX_LAST") - Name of the column
            containing value at date
        load_using_pandas (bool, default:False) - True to use pandas
           package to load the data
    Returns:
        a pandas dataframe consolidating thedataof all files provided
    """
    date_format = "%d/%m/%Y"
    if not all([isinstance(arg, FileDef) for arg in file_defs]):
        raise TypeError("file_defs must be of class FileDef")
    if load_using_pandas:
        df = None
        for file_def in file_defs:
            filename = file_def.filename
            colname = file_def.colname
            temp_df = pd.read_csv(filename)
            temp_df.columns = ["date", colname]
            temp_df[colname] = pd.to_numeric(temp_df[colname])
            temp_df["date"] = pd.to_datetime(temp_df["date"])
            if df is None:
                df = temp_df
            else:
                df = pd.merge(df, temp_df, on="date", how="outer")
    else:
        hash_output = {}  # Keys are date strings, values are dicts
        for file_def in file_defs:
            filename = file_def.filename
            colname = file_def.colname
            with open(filename) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    str_date = row[date_colname]
                    if str_date not in hash_output:
                        hash_output[str_date] = {
                            "date": datetime.datetime.strptime(str_date, date_format)
                        }
                    hash_output[str_date][colname] = float(row[main_colname])
        df = pd.DataFrame(hash_output.values())
    return df


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return end - start

    return wrapper


def pandas_to_heatmap_matrix(
    df: pd.DataFrame, coords_col: str, z_col: str
) -> np.ndarray:
    xs = df[coords_col].apply(lambda val: val[0])
    ys = df[coords_col].apply(lambda val: val[1])

    matrix = np.zeros([max(xs) + 1, max(ys) + 1])
    for _, row in df.iterrows():
        x, y = row[coords_col]
        matrix[x, y] = row[z_col]
    return matrix
