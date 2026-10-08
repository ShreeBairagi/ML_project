import numpy as np
import pandas as pd


def simulate_battery_capacity(
    battery_id: str,
    n_cycles: int = 170,
    initial_capacity: float = 2.0,
    linear_fade: float = 0.0025,
    quadratic_fade: float = 0.000005,
    noise_std: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate a simple battery-capacity degradation trajectory.

    This simulator is intentionally simple and transparent. It is not intended
    to reproduce the NASA batteries exactly.

    Capacity follows a deterministic degradation curve plus Gaussian
    measurement noise.
    """
    if n_cycles < 2:
        raise ValueError("n_cycles must be at least 2")

    if initial_capacity <= 0:
        raise ValueError("initial_capacity must be positive")

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(random_state)

    cycles = np.arange(n_cycles, dtype=int)

    true_capacity = (
        initial_capacity
        - linear_fade * cycles
        - quadratic_fade * cycles**2
    )

    observed_capacity = true_capacity + rng.normal(
        loc=0.0,
        scale=noise_std,
        size=n_cycles,
    )

    return pd.DataFrame(
        {
            "battery_id": battery_id,
            "discharge_cycle_index": cycles,
            "capacity": observed_capacity,
            "true_capacity": true_capacity,
        }
    )


def simulate_battery_population(
    n_batteries: int = 4,
    n_cycles: int = 170,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a small population of batteries with different degradation rates.
    """
    if n_batteries < 1:
        raise ValueError("n_batteries must be at least 1")

    rng = np.random.default_rng(random_state)

    frames = []

    for index in range(n_batteries):
        initial_capacity = rng.uniform(1.9, 2.1)
        linear_fade = rng.uniform(0.0018, 0.0032)
        quadratic_fade = rng.uniform(0.000002, 0.000008)

        frame = simulate_battery_capacity(
            battery_id=f"SYN{index + 1:02d}",
            n_cycles=n_cycles,
            initial_capacity=initial_capacity,
            linear_fade=linear_fade,
            quadratic_fade=quadratic_fade,
            noise_std=0.01,
            random_state=random_state + index + 1,
        )

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)