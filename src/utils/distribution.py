import os
import numpy as np
from bisect import bisect
import json
from matplotlib import pyplot as plt
import seaborn as sns


class Distribution(object):
    def __init__(self, bins=10000):
        self.values = []
        self.quantiles = []
        self.bins = bins

    def add_observation(self, obs: float):
        self.values.append(obs)

    def refresh(self) -> None:
        assert len(self.values) >= 2
        self.values.sort()
        for i in range(self.bins):
            q = i / self.bins
            idxf = q * (len(self.values) - 1)
            idxi = int(idxf)
            threshold = self.values[idxi] + (self.values[idxi + 1] - self.values[idxi]) * (idxf - idxi)
            self.quantiles.append(threshold)

    def pdf(self, obs: float) -> float:  # approximated pdf
        q = self.value2quantile(obs)
        try:
            return 1 / self.bins / (self.quantile2value(q + 1) - self.quantile2value(q))
        except:
            return 0.0

    def mu(self) -> float:
        return sum(self.quantiles) / self.bins
    
    def value2quantile(self, obs: float) -> int:
        return max(0, bisect(self.quantiles, obs) - 1)

    def quantile2value(self, q: int) -> float:
        return self.quantiles[q]
    
    def load(self, path: str):
        with open(path, "r") as f:
            self.quantiles = json.load(f)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.quantiles, f)

    def clear(self):
        self.values = []

    def visualize(self, output_path: str, title="", value_name="Value", xlim=(0, 5)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create figure
        quantiles = self.quantiles
        x = np.linspace(0, 1, self.bins)  # Assuming they are evenly spaced percentiles
        fig, ax1 = plt.subplots()

        # Plot Empirical CDF
        ax1.plot(quantiles, x, label="Empirical CDF", color="blue")
        ax1.set_xlabel(value_name)
        ax1.set_ylabel("Cumulative Probability", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Create a second y-axis for the density function
        ax2 = ax1.twinx()
        sns.kdeplot(quantiles, ax=ax2, color="red", label="Density (PDF)")
        ax2.set_ylabel("Density", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Limit x-axis
        ax1.set_xlim(*xlim)
        ax2.set_xlim(*xlim)

        # Add legends
        ax1.legend(loc="upper right")
        ax2.legend(loc="lower right")

        if title == "":
            title = "Empirical CDF and Density Function"
        plt.title(f"{title}(mu={self.mu():.2f})")
        plt.savefig(output_path)
        plt.close()


def kl_divergence(P: Distribution, Q: Distribution):
    """ Computes KL divergence D_KL(P || Q) using Monte Carlo sampling. """
    xs = P.quantiles  # equivalent to sampling from the distribution P
    P_vals = np.maximum([P.pdf(x) for x in xs], 1e-10)  # Avoid log(0)
    Q_vals = np.maximum([Q.pdf(x) for x in xs], 1e-10)
    return np.mean(np.log(P_vals) - np.log(Q_vals))  # Monte Carlo estimate


if __name__ == "__main__":
    dist = Distribution()
    dist.add_observation(1)
    dist.add_observation(1.2)
    dist.add_observation(100)
    dist.add_observation(101)
    dist.add_observation(101.11)

    dist.refresh()
    print(dist.value2quantile(-100))
    print(dist.value2quantile(35))
    print(dist.value2quantile(2e9))
    print(dist.quantile2value(3))
    print(dist.quantile2value(9999))

    dist.save("./test.json")
    dist.clear()
    dist.load("./test.json")
