from algorithm.utils import load_csv_data, timed
from . import FILE_DEFS

performance_iterations = 10


def test_load_csv_data():
    global FILE_DEFS
    data = load_csv_data(*FILE_DEFS)
    assert data.count()["date"] == 3392
    assert data.count()["spot"] == 3130
    assert data.count()["1y_atmf_vol"] == 3392

    data = load_csv_data(*FILE_DEFS, load_using_pandas=True)
    assert data.count()["date"] == 3392
    assert data.count()["spot"] == 3130
    assert data.count()["1y_atmf_vol"] == 3392


def performance_test_load_csv_data_performance():
    global FILE_DEFS, performance_iterations

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
        using_pandas_times.append(load_using_pandas(*FILE_DEFS))
        without_pandas_times.append(load_without_pandas(*FILE_DEFS))
        iteration += 1

    avg_perf_using_pd = sum(using_pandas_times) / len(using_pandas_times)
    avg_perf_without_pd = sum(without_pandas_times) / len(without_pandas_times)
    print("Avg. perf using Pandas" + str(avg_perf_using_pd) + "s")
    print("Avg. perf without Pandas" + str(avg_perf_without_pd) + "s")
