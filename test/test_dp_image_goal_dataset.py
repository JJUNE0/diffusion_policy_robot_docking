import numpy as np
import h5py
import torch
from PIL import Image

from utils.dp_image_goal_dataset import DPImageGoalDataset
from utils.goal_pool import GoalPool


def _make_goal(pool, tmp_path, dataset, variant, value, episode=0):
    image_path = tmp_path / f"{dataset}_{variant}.png"
    Image.fromarray(np.full((12, 16, 3), value, np.uint8)).save(image_path)
    goal_id = GoalPool.default_goal_id(dataset, "all", episode, variant)
    pool.upsert(
        goal_id=goal_id,
        dataset=dataset,
        split="all",
        episode=episode,
        variant=variant,
        images={"orbbec-0": image_path},
    )


def _dataset(tmp_path):
    h5_path = tmp_path / "dock.h5"
    n = 24
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("episode_ends", data=np.asarray([12, 24], np.int64))
        action = np.linspace(-0.2, 0.3, n * 2, dtype=np.float32).reshape(n, 2)
        h5.create_dataset("encoder", data=action)
        image = np.arange(n * 3 * 12 * 16, dtype=np.uint8).reshape(n, 3, 12, 16)
        h5.create_dataset("image_bottom", data=image)
        string = h5py.string_dtype("utf-8")
        h5.create_dataset("source_dataset", data=np.asarray(["A", "B"], dtype=object), dtype=string)

    db_path = tmp_path / "goals.sqlite3"
    with GoalPool(db_path, create=True) as pool:
        for dataset, base in (("A", 20), ("B", 100)):
            for offset, variant in enumerate(("original", "qr_removed", "color_changed")):
                _make_goal(pool, tmp_path, dataset, variant, base + offset)

    return DPImageGoalDataset(
        h5_path=str(h5_path),
        goal_pool_db=str(db_path),
        horizon=4,
        obs_horizon=6,
        vision_horizon=3,
        vision_stride=2,
        goal_variant_weights={
            "original": 0.495,
            "qr_removed": 0.495,
            "color_changed": 0.01,
        },
        cross_goal_prob=0.1,
        cross_goal_map={"A": "B", "B": "A"},
    )


def test_official_dp_contract_and_vw_units(tmp_path):
    dataset = _dataset(tmp_path)
    sample = dataset[0]
    assert sample["action"].shape == (4, 2)
    assert sample["action"].dtype == torch.float32
    assert sample["obs"]["wheel_history"].shape == (1, 12)
    for key in (*[f"history_{i}" for i in range(3)], "goal"):
        assert sample["obs"][key].shape == (1, 3, 12, 16)
        assert sample["obs"][key].dtype == torch.uint8
    # Episode-start history is left padded by repeating its first physical row.
    assert torch.equal(sample["obs"]["history_0"], sample["obs"]["history_1"])
    assert torch.allclose(sample["action"], torch.from_numpy(dataset.actions[:4]))


def test_cross_goal_and_variant_probabilities_are_hierarchical(tmp_path):
    dataset = _dataset(tmp_path)
    torch.manual_seed(42)
    n = 10_000
    source = str(dataset.episode_sources[0])
    cross = 0
    color = 0
    for _ in range(n):
        row = dataset.sample_goal_row(0)
        cross += str(dataset.goal_datasets[row]) != source
        color += str(dataset.goal_variants[row]) == "color_changed"
    assert 0.08 < cross / n < 0.12
    assert 0.005 < color / n < 0.02
