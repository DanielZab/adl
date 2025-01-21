"""
A helper module to visualize data using matplotlib
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np

from dataset.containers import DataSet


class MonthYearContainer:
    """
    A container for month and year values. Used for efficient sorting and grouping of data.
    """

    def __init__(self, month: int, year: int):
        self.month = month
        self.year = year

    def __eq__(self, other):
        return self.month == other.month and self.year == other.year

    def __gt__(self, other):
        if self.year != other.year:
            return self.year > other.year
        return self.month > other.month

    def __lt__(self, other):
        if self.year != other.year:
            return self.year < other.year
        return self.month < other.month

    def __str__(self):
        return f"{self.month}/{self.year}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.month, self.year))


class Visualizer:
    @staticmethod
    def create_monthly_bins(data: DataSet):
        """
        Creates a dictionary of month-year containers and the number of data points in that month-year container.
        """
        monthly_bins = {}
        for data_point in data.data_points:
            date = datetime.datetime.fromtimestamp(data_point.t / 1000)
            month = int(date.strftime("%m"))
            year = int(date.strftime("%Y"))
            container = MonthYearContainer(month, year)
            if container not in monthly_bins:
                monthly_bins[container] = 0
            monthly_bins[container] += 1
        monthly_bins = monthly_bins.items()
        monthly_bins = sorted(monthly_bins, key=lambda x: x[0])
        return monthly_bins

    @staticmethod
    def plot_monthly_distribution(data: DataSet):
        """
        Plots the monthly distribution of data points in the dataset.
        """

        bins = Visualizer.create_monthly_bins(data)
        x = [str(x[0]) for x in bins]
        y = [x[1] for x in bins]

        width_per_bin = 1  # Adjust this value to scale
        fig_width = len(y) * width_per_bin
        fig_height = 6  # Keep height fixed or adjust as needed
        plt.figure(figsize=(fig_width, fig_height))

        plt.title("Monthly Data Availability of " + data.name)
        plt.bar(x, y)
        plt.xticks(x, rotation=45, ha="right")

        plt.show()

    @staticmethod
    def plot_price_history(data: DataSet):
        """
        Plots the price history of a dataset.
        """
        x = [datetime.datetime.fromtimestamp(x.t / 1000) for x in data.data_points]
        y = [x.c for x in data.data_points]

        plt.figure(figsize=(12, 6))
        plt.title("Price History of " + data.name)
        plt.plot(x, y)
        plt.show()

    @staticmethod
    def plot_traces(traces: list):

        """
        Plots data related to the traces of the agent. Plotted are rewards, actions, the current price, the balance, the number of stocks owned, and the portfolio value.
        """

        # Create a 2x4 grid of subplots
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

        # Titles for each subplot
        titles = [
            "Rewards",
            "Actions",
            "current price",
            "balance",
            "stocks_owned",
            "portfolio_value",
        ]

        # Flatten the axes array to make it easier to iterate
        axes = axes.flatten()

        for nr, states, actions, rewards in traces:
            x = rewards
            axes[0].plot(x, label=f"Version {nr}")

            x = actions
            axes[1].plot(x, label=f"Version {nr}")

            x = [e[0]["current_price"] for e in states]
            axes[2].plot(x, label=f"Version {nr}")

            x = [e[0]["balance"] for e in states]
            axes[3].plot(x, label=f"Version {nr}")

            x = [e[0]["stocks_owned"] for e in states]
            axes[4].plot(x, label=f"Version {nr}")

            x = [e[1]["portfolio_value"] for e in states]
            axes[5].plot(x, label=f"Version {nr}")

        # Set legend and titles
        axes[-1].legend()
        res = [axes[i].set_title(t) for i, t in enumerate(titles)]

        plt.show()
