import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from algorithm.stat_methods import gridiserFactory


class PandasHeatMapPlot:
    """Object to plot a heatmap of the hit rates.
    The hit rates are calculated by grouping rows in a pandas
    dataframe in a specified number of cells. All the rows
    that fall inside each cell are counted. The number of positive
    counts from a third boolean column are aggregated at each cell.
    The hit rate on each cell is the proportion of positive counts
    to total counts."""

    def __init__(
        self,
        df: pd.DataFrame,
        xdivs: int,
        ydivs: int,
        xcolname: str,
        ycolname: str,
        pcolname: str,
    ) -> None:
        """Args:
        df: a dataframe with the following columns:
            x (values of x-axis), y (values of x-axis),
            p (a column with bool values to aggregate).
        xcolname: the name of the col with the values of the x-axis
        ycolname: the name of the col with the values of the y-axis
        pcolname: the name of the column with bool values
            coordinates
        """
        self._xdivs = xdivs
        self._ydivs = ydivs
        self._grouped_df = self._data_to_grid(
            df, xdivs, ydivs, xcolname, ycolname, pcolname
        )
        self._heat_matrix = self._pandas_to_heatmap_matrix(
            self._grouped_df, "coordinates", "hit_rate"
        )
        self._min_x = df[xcolname].min()
        self._max_x = df[xcolname].max()
        self._min_y = df[ycolname].min()
        self._max_y = df[ycolname].max()

    @classmethod
    def _data_to_grid(
        cls,
        df: pd.DataFrame,
        xdivs: float,
        ydivs: float,
        xcolname: str,
        ycolname: str,
        pcolname: str,
    ) -> pd.DataFrame:
        shape = (
            {"divisions": xdivs, "max": df[xcolname].max(), "min": df[xcolname].min()},
            {"divisions": ydivs, "max": df[ycolname].max(), "min": df[ycolname].min()},
        )
        gridise = gridiserFactory(shape)
        df["cell"] = [
            str(gridise(row[xcolname], row[ycolname])) for _, row in df.iterrows()
        ]
        return cls._group_dataframe(df, xcolname, ycolname, pcolname)

    @classmethod
    def _group_dataframe(
        cls,
        df: pd.DataFrame,
        xcolname: str,
        ycolname: float,
        pcolname: str,
    ) -> pd.DataFrame:
        # TODO: Perform grouping without pandas and make more efficient
        grouping_cols = ["cell", xcolname, ycolname, pcolname]
        aggreg_df = df[grouping_cols]

        avg_df = pd.pivot_table(
            aggreg_df,
            index=["cell"],
            values=[xcolname, ycolname],
            aggfunc=np.average,
        )
        count_df = pd.pivot_table(
            aggreg_df,
            index=["cell"],
            values=pcolname,
            aggfunc=len,
        )
        count_df.columns = ["total_count"]

        sum_df = pd.pivot_table(
            aggreg_df,
            index=["cell"],
            values=pcolname,
            aggfunc=sum,
        )
        sum_df.columns = ["positive_count"]

        grouped_df = pd.concat([avg_df, count_df, sum_df], axis=1)
        grouped_df["hit_rate"] = grouped_df.positive_count / grouped_df.total_count
        grouped_df["coordinates"] = [cls._parse_tuple(val) for val in grouped_df.index]
        return grouped_df

    @staticmethod
    def _pandas_to_heatmap_matrix(
        df: pd.DataFrame, coords_col: str, z_col: str
    ) -> np.ndarray:
        xs = df[coords_col].apply(lambda val: val[0])
        ys = df[coords_col].apply(lambda val: val[1])

        matrix = np.zeros([max(xs) + 1, max(ys) + 1])
        for _, row in df.iterrows():
            x, y = row[coords_col]
            matrix[x, y] = row[z_col]
        return matrix

    @staticmethod
    def _parse_tuple(val) -> tuple:
        return tuple(
            int(num)
            for num in val.replace("'", "")
            .replace('"', "")
            .replace("(", "")
            .replace(")", "")
            .split(",")
        )

    def show(self, xlabel: str = "x", ylabel: str = "y") -> None:
        """Show the created plot.

        Args:
            xlabel: label for the x-axis
            ylabel: label for the y-axis
        """
        # TODO: Improve plot presentation
        indexes = np.round(np.linspace(self._min_x, self._max_x, self._xdivs) * 100)
        columns = np.round(np.linspace(self._min_y, self._max_y, self._ydivs), 4) * 100
        plottable_df = pd.DataFrame(self._heat_matrix, columns=columns, index=indexes)
        plottable_df = plottable_df[np.sort(columns)[::-1]]
        ax = sns.heatmap(plottable_df.T)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        plt.show()
