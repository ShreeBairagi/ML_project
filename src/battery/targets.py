import pandas as pd


def add_next_cycle_capacity_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the next-cycle capacity target for one battery.

    At cycle t:
    input information is available through cycle t
    target is capacity at cycle t + 1

    The final discharge cycle has no next-cycle target and is removed.
    """
    required_columns = {
        "battery_id",
        "discharge_cycle_index",
        "capacity",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.sort_values(
        ["battery_id", "discharge_cycle_index"]
    ).copy()

    result["target_next_capacity"] = (
        result.groupby("battery_id")["capacity"].shift(-1)
    )

    result = result.dropna(
        subset=["target_next_capacity"]
    ).reset_index(drop=True)

    return result