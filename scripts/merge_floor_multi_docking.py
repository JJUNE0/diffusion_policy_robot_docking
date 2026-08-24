#!/usr/bin/env python3.12
"""Physically merge floor4_hallway/floor4_inside/floor5 front-dock datasets
(first N_EPISODES each = the designated training split) into ONE h5 matching
after_0328_train.h5's row-schema, so it can go through the same
scripts/precompute_reloc3r_cache.py + precompute_reloc3r_dec_features.py
pipeline and then train with the existing r_relfeat sensors_variant configs.

Only the 4 keys common to all three raw sources are copied: `encoder`,
`episode_ends`, `image_bottom`, `image_top`. Each source is trimmed to its
first N_EPISODES episodes (the pre-labeled/pre-named "trainset" portion --
see floor4_hallway_front_docking.h5's own `selection_description` attr and
the `Dataset_episode_num_80_trainset` source-path convention shared by all
three) before concatenation, so validation/held-out/known-bad episodes never
enter the merged file.

Usage: python scripts/merge_floor_multi_docking.py
"""
import h5py
import numpy as np

N_EPISODES = 80
ROW_BLOCK = 1024  # streamed copy chunk, bounds peak RAM for the big image arrays

SOURCES = [
    "/home/work/.postech/jiwon/diffusion_policy_robot_docking_jiwon/dataset/floor4_hallway_front_docking.h5",
    "/home/work/.postech/jiwon/diffusion_policy_robot_docking_jiwon/dataset/floor4_inside_front_docking.h5",
    "/home/work/.postech/jiwon/diffusion_policy_robot_docking_jiwon/dataset/floor5_front_docking.h5",
]
OUT_PATH = "/home/work/.postech/diffusion_policy_robot_docking/dataset/floor_multi_train.h5"


def main():
    # ---- Pass 1: figure out per-source row cutoffs + combined shapes ----
    cutoffs = []
    total_rows = 0
    total_episodes = 0
    for path in SOURCES:
        with h5py.File(path, "r") as f:
            ends = f["episode_ends"][:]
            if len(ends) < N_EPISODES:
                raise ValueError(f"{path}: only {len(ends)} episodes, need >= {N_EPISODES}")
            cutoff = int(ends[N_EPISODES - 1])
            cutoffs.append(cutoff)
            total_rows += cutoff
            total_episodes += N_EPISODES
            print(f"{path}: using rows 0..{cutoff} ({N_EPISODES} episodes)")

    print(f"TOTAL merged rows={total_rows} episodes={total_episodes}")

    with h5py.File(SOURCES[0], "r") as f0:
        img_dtype = f0["image_bottom"].dtype
        img_hw = f0["image_bottom"].shape[1:]  # (3, 240, 320)
        enc_dtype = f0["encoder"].dtype
        enc_dim = f0["encoder"].shape[1]

    # ---- Create output file ----
    with h5py.File(OUT_PATH, "w") as out:
        out.create_dataset("encoder", shape=(total_rows, enc_dim), dtype=enc_dtype)
        out.create_dataset("episode_ends", shape=(total_episodes,), dtype=np.int64)
        for key in ("image_bottom", "image_top"):
            out.create_dataset(
                key, shape=(total_rows, *img_hw), dtype=img_dtype,
                chunks=(1, *img_hw), compression="gzip",
            )
        # provenance, so we can trace a merged row back to its source dataset
        out.create_dataset("source_dataset", shape=(total_rows,), dtype=h5py.string_dtype())
        out.create_dataset("source_row", shape=(total_rows,), dtype=np.int64)

        row_offset = 0
        ep_offset = 0
        all_ends = []
        for path, cutoff in zip(SOURCES, cutoffs):
            name = path.rsplit("/", 1)[-1].removesuffix(".h5")
            with h5py.File(path, "r") as f:
                ends = f["episode_ends"][:N_EPISODES]
                all_ends.append(ends.astype(np.int64) + row_offset)

                out["encoder"][row_offset:row_offset + cutoff] = f["encoder"][:cutoff]
                out["source_dataset"][row_offset:row_offset + cutoff] = name
                out["source_row"][row_offset:row_offset + cutoff] = np.arange(cutoff, dtype=np.int64)

                for start in range(0, cutoff, ROW_BLOCK):
                    stop = min(start + ROW_BLOCK, cutoff)
                    out["image_bottom"][row_offset + start:row_offset + stop] = f["image_bottom"][start:stop]
                    out["image_top"][row_offset + start:row_offset + stop] = f["image_top"][start:stop]
                    if start % (ROW_BLOCK * 20) == 0:
                        print(f"  {name}: {stop}/{cutoff} rows copied")

            print(f"{name}: done, {cutoff} rows -> merged rows [{row_offset}, {row_offset + cutoff})")
            row_offset += cutoff
            ep_offset += N_EPISODES

        out["episode_ends"][:] = np.concatenate(all_ends)
        out.attrs["merged_from"] = SOURCES
        out.attrs["episodes_per_source"] = N_EPISODES

    print(f"Wrote {OUT_PATH}")
    with h5py.File(OUT_PATH, "r") as f:
        print("Verify: rows=", f["encoder"].shape[0], "episodes=", f["episode_ends"].shape[0])
        print("episode_ends tail per source boundary:",
              [int(f["episode_ends"][i * N_EPISODES - 1]) for i in range(1, len(SOURCES) + 1)])


if __name__ == "__main__":
    main()
