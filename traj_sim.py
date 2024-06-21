import numpy as np


def generate_real_samples(n):
        epsilon: float = 1e-3           # Inverse temperature.
        num_steps: int = n           # Number of steps to integrate.
        dim: int = 2                    # Spatial dimensions.
        dt: float = 1                # Time step length.
        def vector_field(x: np.ndarray) -> np.ndarray:
            return np.array([1e-3, (1-x[1])*1e-1])
        scale_param = np.array([1e-3, 1e-1])
        random_number_generator = np.random.default_rng()
        xs = np.zeros((num_steps, dim))
        for i in range(1, num_steps):
            x_prev = xs[i-1, :]
            noise = random_number_generator.normal(scale=scale_param, size=2)
            xs[i, :] = x_prev + vector_field(x_prev) * dt + noise
        X1 = xs[:, 0]
        X2 = xs[:, 1]
        X = xs
        labels = X1
        return X, labels