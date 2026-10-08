import pandas as pd

from src.battery.splits import (
    leave_one_battery_out,
    random_cycle_split_wrong,
    time_ordered_within_battery,
)


def make_test_data():
    rows = []

    for battery_id in ["B1", "B2"]:
        for cycle in range(10):
            rows.append(
                {
                    "battery_id": battery_id,
                    "discharge_cycle_index": cycle,
                    "capacity": 2.0 - cycle * 0.01,
                }
            )

    return pd.DataFrame(rows)


def test_leave_one_battery_out():
    df = make_test_data()

    train_df, test_df = leave_one_battery_out(df, "B2")

    assert set(train_df["battery_id"]) == {"B1"}
    assert set(test_df["battery_id"]) == {"B2"}

    assert set(train_df["battery_id"]).isdisjoint(
        set(test_df["battery_id"])
    )


def test_random_cycle_split_preserves_all_rows():
    df = make_test_data()

    train_df, test_df = random_cycle_split_wrong(
        df,
        test_size=0.2,
        random_state=42,
    )

    assert len(train_df) + len(test_df) == len(df)


def test_time_ordered_split():
    df = make_test_data()

    train_df, test_df = time_ordered_within_battery(
        df,
        train_fraction=0.8,
    )

    for battery_id in ["B1", "B2"]:
        train_cycles = train_df[
            train_df["battery_id"] == battery_id
        ]["discharge_cycle_index"]

        test_cycles = test_df[
            test_df["battery_id"] == battery_id
        ]["discharge_cycle_index"]

        assert train_cycles.max() < test_cycles.min()