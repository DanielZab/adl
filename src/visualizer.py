import datetime
import matplotlib.pyplot as plt

from dataset_container import DataSet

class MonthYearContainer:
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

        bins = Visualizer.create_monthly_bins(data)
        x = [str(x[0]) for x in bins]
        y = [x[1] for x in bins]

        width_per_bin = 1  # Adjust this value to scale
        fig_width = len(y) * width_per_bin
        fig_height = 6  # Keep height fixed or adjust as needed
        plt.figure(figsize=(fig_width, fig_height))

        plt.title("Monthly Data Availability of " + data.name)
        plt.bar(x, y)
        plt.xticks(x, rotation=45, ha='right')

        plt.show()

    @staticmethod
    def plot_price_history(data: DataSet):
        x = [datetime.datetime.fromtimestamp(x.t / 1000) for x in data.data_points]
        y = [x.c for x in data.data_points]

        plt.figure(figsize=(12, 6))
        plt.title("Price History of " + data.name)
        plt.plot(x, y)
        plt.show()