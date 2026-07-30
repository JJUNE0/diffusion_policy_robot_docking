import h5py
import numpy as np
import torch

from utils.modular_dataset import ModularDockingDataset


def _strings(values):
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def test_variant_weights_balance_categories_not_rows(tmp_path):
    data_path = tmp_path / "data.h5"
    pool_path = tmp_path / "goals.h5"
    with h5py.File(data_path, "w") as h:
        h.create_dataset("episode_ends", data=np.asarray([3], np.int64))
        h.create_dataset("encoder", data=np.zeros((3, 2), np.float32))
        h.create_dataset("history", data=np.zeros((3, 1, 1), np.float16))
        h.create_dataset("source_dataset", data=_strings(["floor4"]))

    variants = ["original"] * 10 + ["qr_removed"] * 10 + ["color_changed"]
    codes = {"original": 1, "qr_removed": 2, "color_changed": 3}
    with h5py.File(pool_path, "w") as h:
        h.create_dataset("goal_id", data=_strings([f"g{i}" for i in range(21)]))
        h.create_dataset("dataset", data=_strings(["floor4"] * 21))
        h.create_dataset("split", data=_strings(["all"] * 21))
        h.create_dataset("variant", data=_strings(variants))
        h.create_dataset(
            "feat",
            data=np.asarray([[[codes[v]]] for v in variants], np.float16),
        )

    sensors = {
        "pair": {
            "encoder": "reloc3r_goal_pair",
            "source": "history",
            "mode": "goal_pool_pair",
            "horizon": 1,
            "stride": 1,
            "goal_pool_file": str(pool_path),
            "goal_source": "feat",
            "goal_pool_group": "paired",
            "goal_pool_filter": {"split": ["all"]},
            "goal_sampling": "random",
            "goal_variant_weights": {
                "original": 1.0,
                "qr_removed": 1.0,
                "color_changed": 1.0,
            },
            "goal_pool_match": {
                "episode_field": "source_dataset",
                "goal_field": "dataset",
                "cross_probability": 0.0,
            },
        }
    }
    ds = ModularDockingDataset(
        h5_path=str(data_path), sensors=sensors, horizon=1, obs_horizon=1,
        action_key="encoder", train_h5_path=str(data_path), action_norm="minmax",
    )
    torch.manual_seed(7)
    counts = {1: 0, 2: 0, 3: 0}
    for _ in range(6000):
        code = int(ds[0]["obs"]["pair"][0, 1, 0, 0].item())
        counts[code] += 1
    for count in counts.values():
        assert 1700 < count < 2300, counts
