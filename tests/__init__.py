import os
from algorithm.utils import FileDef

MARKET_DATA_DIR = os.path.join("tests", "test_data")
SPOT_DATA_FILE = os.path.join(MARKET_DATA_DIR, "EURUSDxSPOT.csv")
VOL_DATA_FILE = os.path.join(MARKET_DATA_DIR, "EURUSDxVOL.csv")

FILE_DEFS = [
    FileDef(filename=SPOT_DATA_FILE, colname="spot"),
    FileDef(filename=VOL_DATA_FILE, colname="1y_atmf_vol"),
]
