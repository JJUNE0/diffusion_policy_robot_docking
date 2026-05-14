"""Resolve checkpoint paths for Hydra inference scripts."""

from pathlib import Path


def resolve_inference_checkpoint_path(args, original_cwd: str) -> str:
    """
    Build an absolute path to a .pt checkpoint.

    Priority:
      1) ``checkpoint_path`` — full or repo-relative path to a .pt file (overrides dir+step)
      2) ``checkpoint_dir`` + ``checkpoint_step`` → ``checkpoint_dir/checkpoint_step_{step}.pt``
      3) If ``checkpoint_dir`` is unset, ``original_cwd/checkpoint_step_{step}.pt`` (legacy)

    Paths that are not absolute are joined with ``original_cwd``.
    """
    original = Path(original_cwd)

    cp = args.get("checkpoint_path")
    if cp not in (None, "", "null"):
        p = Path(str(cp).strip())
        if not p.is_absolute():
            p = original / p
        return str(p)

    step = args.get("checkpoint_step")
    if step is None:
        raise ValueError("Either checkpoint_path or checkpoint_step must be set.")

    ckpt_dir = args.get("checkpoint_dir")
    if ckpt_dir not in (None, "", "null"):
        dir_path = Path(str(ckpt_dir).strip())
        if not dir_path.is_absolute():
            dir_path = original / dir_path
    else:
        dir_path = original

    fname = f"checkpoint_step_{int(step)}.pt"
    return str(dir_path / fname)
