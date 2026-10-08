import pandas as pd
from sklearn.model_selection import train_test_split


def leave_one_battery_out(df: pd.DataFrame, held_out_battery: str):
    """
    Primary evaluation split.

    Train on all batteries except held_out_battery.
    Test on held_out_battery only.
    """
    train_df = df[df["battery_id"] != held_out_battery].copy()
    test_df = df[df["battery_id"] == held_out_battery].copy()

    if train_df.empty:
        raise ValueError("LOBO training set is empty.")

    if test_df.empty:
        raise ValueError(
            f"Held-out battery {held_out_battery} not found."
        )

    return train_df, test_df


def random_cycle_split_wrong(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Deliberately WRONG split used only for the leakage demonstration.

    Cycles from the same battery may appear in both train and test sets.
    Never use this as the primary reported evaluation.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def time_ordered_within_battery(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
):
    """
    Secondary temporal evaluation.

    For each battery:
    - earlier cycles go to training
    - later cycles go to testing
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    train_parts = []
    test_parts = []

    for battery_id, battery_df in df.groupby("battery_id"):
        battery_df = battery_df.sort_values(
            "discharge_cycle_index"
        ).reset_index(drop=True)

        if len(battery_df) < 2:
            raise ValueError(
                f"Battery {battery_id} needs at least 2 rows."
            )

        split_index = int(len(battery_df) * train_fraction)

        if split_index == 0 or split_index == len(battery_df):
            raise ValueError(
                f"Invalid time split for battery {battery_id}."
            )

        train_parts.append(
            battery_df.iloc[:split_index].copy()
        )

        test_parts.append(
            battery_df.iloc[split_index:].copy()
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    return train_df, test_df