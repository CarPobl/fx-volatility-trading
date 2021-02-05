from . import SPOT_DATA_FILE
from . import VOL_DATA_FILE

from utils import load_csv_data, FileDef


# Load Market data
file_defs = [
    FileDef(filename=SPOT_DATA_FILE, colname="spot"),
    FileDef(filename=VOL_DATA_FILE, colname="1y_atmf_vol"),
]
market_data = load_csv_data(*file_defs)


