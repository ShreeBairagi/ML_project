import pandas as pd

from src.battery.targets import add_next_cycle_capacity_target


def test_next_cycle_capacity_target():
    df = pd.DataFrame(
        {
            "battery_id": ["BTEST"] * 4,
            "discharge_cycle_index": [0, 1, 2, 3],
            "capacity": [2.0, 1.9, 1.8, 1.7],
        }
    )

    result = add_next_cycle_capacity_target(df)

    assert len(result) == 3
    assert result["target_next_capacity"].tolist() == [1.9, 1.8, 1.7]

    assert result["discharge_cycle_index"].tolist() == [0, 1, 2]