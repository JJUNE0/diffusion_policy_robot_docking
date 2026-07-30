"""Editable, camera-paired goal-image pool.

The SQLite database is the source of truth.  Image bytes live next to it under
``images/`` so entries can be added, replaced, disabled, restored, and audited
without rewriting a monolithic HDF5 file.  ``scripts/compile_goal_pool.py``
turns the enabled rows into the immutable ReLoc3R encoder-feature snapshot used
by training.

A *goal* is a synchronized set of camera images.  Sampling is therefore done
by goal id, never independently per camera.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_component(value: str) -> str:
    out = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))
    return out.strip("._") or "unnamed"


@dataclass(frozen=True)
class GoalRecord:
    goal_id: str
    dataset: str
    split: str
    episode: int
    variant: str
    enabled: bool
    parent_id: Optional[str]
    note: str
    metadata: dict
    images: Dict[str, dict]


class GoalPool:
    def __init__(self, db_path: str | os.PathLike, create: bool = False):
        self.db_path = Path(db_path).expanduser().resolve()
        self.root = self.db_path.parent
        self.image_root = self.root / "images"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            self.image_root.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists() and not create:
            raise FileNotFoundError(f"goal pool database does not exist: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if create:
            self._init_schema()
        self._check_schema()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                split TEXT NOT NULL,
                episode INTEGER NOT NULL,
                variant TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
                parent_id TEXT REFERENCES goals(goal_id),
                note TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(dataset, split, episode, variant)
            );
            CREATE TABLE IF NOT EXISTS goal_images (
                goal_id TEXT NOT NULL REFERENCES goals(goal_id) ON DELETE CASCADE,
                camera TEXT NOT NULL,
                relpath TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(goal_id, camera)
            );
            CREATE INDEX IF NOT EXISTS idx_goals_enabled
                ON goals(enabled, split, dataset, variant);
            """
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def _check_schema(self):
        try:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise ValueError(f"{self.db_path} is not a goal-pool database") from exc
        if row is None or int(row["value"]) != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported goal-pool schema in {self.db_path}: "
                f"expected {SCHEMA_VERSION}, got {None if row is None else row['value']}"
            )

    @staticmethod
    def default_goal_id(dataset: str, split: str, episode: int, variant: str) -> str:
        return (
            f"{_safe_component(dataset)}:{_safe_component(split)}:"
            f"ep{int(episode):04d}:{_safe_component(variant)}"
        )

    def _copy_image(self, goal_id: str, camera: str, source: Path) -> dict:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            suffix = source.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                suffix = ".png"
            rel = Path(_safe_component(goal_id)) / f"{_safe_component(camera)}{suffix}"
            dst = self.image_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", dir=str(dst.parent))
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                if suffix == source.suffix.lower() and im.mode == "RGB":
                    shutil.copyfile(source, tmp)
                else:
                    fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[suffix]
                    rgb.save(tmp, format=fmt)
                os.replace(tmp, dst)
            finally:
                if tmp.exists():
                    tmp.unlink()
        return {
            "relpath": str(Path("images") / rel),
            "sha256": _sha256(dst),
            "width": width,
            "height": height,
        }

    def upsert(
        self,
        *,
        goal_id: str,
        dataset: str,
        split: str,
        episode: int,
        variant: str,
        images: Dict[str, str | os.PathLike],
        enabled: bool = True,
        parent_id: Optional[str] = None,
        note: str = "",
        metadata: Optional[dict] = None,
        replace: bool = False,
    ) -> str:
        existing = self.conn.execute(
            "SELECT goal_id FROM goals WHERE goal_id=?", (goal_id,)
        ).fetchone()
        if existing and not replace:
            raise FileExistsError(
                f"goal '{goal_id}' already exists; use update/--replace explicitly"
            )
        copied = {
            str(camera): self._copy_image(goal_id, str(camera), Path(path))
            for camera, path in images.items()
        }
        if not copied:
            raise ValueError("at least one --camera NAME=PATH image is required")
        payload = json.dumps(metadata or {}, sort_keys=True)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO goals(
                    goal_id,dataset,split,episode,variant,enabled,parent_id,note,metadata_json
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(goal_id) DO UPDATE SET
                    dataset=excluded.dataset, split=excluded.split,
                    episode=excluded.episode, variant=excluded.variant,
                    enabled=excluded.enabled, parent_id=excluded.parent_id,
                    note=excluded.note, metadata_json=excluded.metadata_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    goal_id,
                    dataset,
                    split,
                    int(episode),
                    variant,
                    int(enabled),
                    parent_id,
                    note,
                    payload,
                ),
            )
            for camera, info in copied.items():
                self.conn.execute(
                    """
                    INSERT INTO goal_images(goal_id,camera,relpath,sha256,width,height)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(goal_id,camera) DO UPDATE SET
                        relpath=excluded.relpath, sha256=excluded.sha256,
                        width=excluded.width, height=excluded.height,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        goal_id,
                        camera,
                        info["relpath"],
                        info["sha256"],
                        info["width"],
                        info["height"],
                    ),
                )
        return goal_id

    def update_images(
        self, goal_id: str, images: Dict[str, str | os.PathLike]
    ):
        row = self.conn.execute(
            "SELECT goal_id FROM goals WHERE goal_id=?", (goal_id,)
        ).fetchone()
        if row is None:
            raise KeyError(goal_id)
        copied = {
            str(camera): self._copy_image(goal_id, str(camera), Path(path))
            for camera, path in images.items()
        }
        with self.conn:
            for camera, info in copied.items():
                self.conn.execute(
                    """
                    INSERT INTO goal_images(goal_id,camera,relpath,sha256,width,height)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(goal_id,camera) DO UPDATE SET
                        relpath=excluded.relpath, sha256=excluded.sha256,
                        width=excluded.width, height=excluded.height,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        goal_id,
                        camera,
                        info["relpath"],
                        info["sha256"],
                        info["width"],
                        info["height"],
                    ),
                )
            self.conn.execute(
                "UPDATE goals SET updated_at=CURRENT_TIMESTAMP WHERE goal_id=?",
                (goal_id,),
            )

    def set_enabled(self, goal_id: str, enabled: bool):
        with self.conn:
            cur = self.conn.execute(
                "UPDATE goals SET enabled=?,updated_at=CURRENT_TIMESTAMP WHERE goal_id=?",
                (int(enabled), goal_id),
            )
        if cur.rowcount != 1:
            raise KeyError(goal_id)

    def records(
        self,
        *,
        enabled: Optional[bool] = None,
        datasets: Optional[Iterable[str]] = None,
        splits: Optional[Iterable[str]] = None,
        variants: Optional[Iterable[str]] = None,
    ) -> List[GoalRecord]:
        clauses, args = [], []
        for column, values in (
            ("dataset", datasets),
            ("split", splits),
            ("variant", variants),
        ):
            values = list(values or [])
            if values:
                clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
                args.extend(values)
        if enabled is not None:
            clauses.append("enabled=?")
            args.append(int(enabled))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            "SELECT * FROM goals" + where + " ORDER BY dataset,split,episode,variant",
            args,
        ).fetchall()
        out = []
        for row in rows:
            image_rows = self.conn.execute(
                "SELECT * FROM goal_images WHERE goal_id=? ORDER BY camera",
                (row["goal_id"],),
            ).fetchall()
            images = {
                ir["camera"]: {
                    "path": str((self.root / ir["relpath"]).resolve()),
                    "relpath": ir["relpath"],
                    "sha256": ir["sha256"],
                    "width": int(ir["width"]),
                    "height": int(ir["height"]),
                }
                for ir in image_rows
            }
            out.append(
                GoalRecord(
                    goal_id=row["goal_id"],
                    dataset=row["dataset"],
                    split=row["split"],
                    episode=int(row["episode"]),
                    variant=row["variant"],
                    enabled=bool(row["enabled"]),
                    parent_id=row["parent_id"],
                    note=row["note"],
                    metadata=json.loads(row["metadata_json"]),
                    images=images,
                )
            )
        return out

    def validate(
        self, required_cameras: Iterable[str] = (), *, enabled_only: bool = False
    ) -> List[str]:
        required = set(required_cameras)
        errors: List[str] = []
        for rec in self.records(enabled=True if enabled_only else None):
            missing = sorted(required - set(rec.images))
            if missing:
                errors.append(f"{rec.goal_id}: missing cameras {missing}")
            for camera, info in rec.images.items():
                path = Path(info["path"])
                if not path.is_file():
                    errors.append(f"{rec.goal_id}/{camera}: missing file {path}")
                    continue
                if _sha256(path) != info["sha256"]:
                    errors.append(f"{rec.goal_id}/{camera}: sha256 mismatch")
                try:
                    with Image.open(path) as im:
                        if im.size != (info["width"], info["height"]):
                            errors.append(
                                f"{rec.goal_id}/{camera}: dimensions changed "
                                f"{im.size} != {(info['width'], info['height'])}"
                            )
                except Exception as exc:
                    errors.append(f"{rec.goal_id}/{camera}: unreadable image: {exc}")
        return errors
