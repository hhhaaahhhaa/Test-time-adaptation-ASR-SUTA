import os
import numpy as np

from .distribution import Distribution


def generate_distribution(tag: str) -> Distribution:
    cache_path = f"results/benchmark/ot/_cache/{tag}.json"    
    print(f"Collect distribution from tag {tag}, cached at {cache_path}.")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        distribution = Distribution()
        distribution.load(cache_path)
        return distribution
    
    # No cache
    if tag == "debug":
        distribution = CutGaussian(0.25, 0.25)
    else:
        raise NotImplementedError
    
    distribution.save(cache_path)
    distribution.visualize(cache_path[:-5] + ".jpg", title=cache_path[:-5], value_name="Entropy")
    print(f"Saved distribution to {cache_path}.")

    return distribution


class CutGaussian(Distribution):
    def __init__(self, mu: float=0.0, sigma: float=1.0):
        super().__init__()
        noise = sigma * np.random.randn(10000) + mu
        for x in noise:
            if x <= 0:
                continue
            self.add_observation(x)
        self.refresh()
