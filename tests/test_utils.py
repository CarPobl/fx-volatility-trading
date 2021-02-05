from utils import load_csv_data, FileDef, timed
from . import SPOT_DATA_FILE
from . import VOL_DATA_FILE

file_defs = [
    FileDef(filename=SPOT_DATA_FILE, colname="spot"),
    FileDef(filename=VOL_DATA_FILE, colname="1y_atmf_vol"),
]

performance_iterations = 10


def test_load_csv_data():
    global file_defs
    data = load_csv_data(*file_defs)
    assert data.count()["date"] == 3392
    assert data.count()["spot"] == 3130
    assert data.count()["1y_atmf_vol"] == 3392

    data = load_csv_data(*file_defs, load_using_pandas=True)
    assert data.count()["date"] == 3392
    assert data.count()["spot"] == 3130
    assert data.count()["1y_atmf_vol"] == 3392


def test_load_csv_data_performance():
    global file_defs, performance_iterations
    @timed
    def load_using_pandas(*args):
        return load_csv_data(*args, load_using_pandas=True)
    @timed
    def load_without_pandas(*args):
        return load_csv_data(*args, load_using_pandas=False)
    
    using_pandas_times = []
    without_pandas_times = []
    
    iteration = 0
    while iteration < performance_iterations:
        using_pandas_times.append(load_using_pandas(*file_defs))
        without_pandas_times.append(load_without_pandas(*file_defs))
        iteration += 1
    
    avg_perf_using_pd = sum(using_pandas_times) / len(using_pandas_times)
    avg_perf_without_pd = sum(without_pandas_times) / len(without_pandas_times)
    print("Avg. perf using Pandas" + str(avg_perf_using_pd) + "s")
    print("Avg. perf without Pandas" + str(avg_perf_without_pd) + "s")