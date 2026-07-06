#!/usr/bin/env python3
"""Split a huge file into fixed-size chunks for cloud upload, then rejoin + verify.

이 저장소의 after_0328_train.h5(~63GB)처럼 큰 파일을 클라우드에 올릴 때 사용한다.
h5 내부 데이터가 이미 압축돼 있어(gzip 해도 ~1%만 줄어듦) **압축은 하지 않고 분할만** 한다.
청크마다 + 원본 전체의 sha256 을 manifest.json 에 기록하므로, 업로드 중 손상/누락을
재조립 시점에 확실히 잡아낸다.

사용법
------
로컬(분할):
    python scripts/chunk_transfer.py split dataset/after_0328_train.h5 \
        --out-dir upload_chunks --chunk-size 10G

    -> upload_chunks/ 에 after_0328_train.h5.part000, .part001, ... 와 manifest.json 생성.
       이 폴더를 통째로 클라우드에 업로드한다.

클라우드(재조립 + 검증):
    python scripts/chunk_transfer.py join upload_chunks \
        --out dataset/after_0328_train.h5

    -> 폴더 안의 .part* 조각들을 glob 으로 긁어 순서대로 이어붙이고 sha256 을 검증한다.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

READ_BLOCK = 64 * 1024 * 1024  # 64MB streaming block (low RAM)


def parse_size(s: str) -> int:
    """'10G', '500M', '1024K', '1048576' -> bytes."""
    s = str(s).strip().upper()
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if s and s[-1] in units:
        return int(float(s[:-1]) * units[s[-1]])
    return int(s)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def split(src: str, out_dir: str, chunk_size: int) -> None:
    if not os.path.isfile(src):
        sys.exit(f"[error] source not found: {src}")
    os.makedirs(out_dir, exist_ok=True)

    total = os.path.getsize(src)
    base = os.path.basename(src)
    n_chunks = (total + chunk_size - 1) // chunk_size
    print(f"splitting {base} ({human(total)}) -> {n_chunks} chunk(s) of {human(chunk_size)} in {out_dir}/")

    full_hash = hashlib.sha256()
    chunks = []
    with open(src, "rb") as fin:
        for idx in range(n_chunks):
            part_name = f"{base}.part{idx:03d}"
            part_path = os.path.join(out_dir, part_name)
            part_hash = hashlib.sha256()
            written = 0
            with open(part_path, "wb") as fout:
                while written < chunk_size:
                    to_read = min(READ_BLOCK, chunk_size - written)
                    block = fin.read(to_read)
                    if not block:
                        break
                    fout.write(block)
                    full_hash.update(block)
                    part_hash.update(block)
                    written += len(block)
            chunks.append({"file": part_name, "size": written, "sha256": part_hash.hexdigest()})
            print(f"  [{idx + 1}/{n_chunks}] {part_name}  {human(written)}")

    manifest = {
        "name": base,
        "size": total,
        "sha256": full_hash.hexdigest(),
        "chunk_size": chunk_size,
        "chunks": chunks,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done. sha256={manifest['sha256']}")
    print(f"manifest -> {os.path.join(out_dir, 'manifest.json')}")


def join(in_dir: str, out_path: str | None) -> None:
    if not os.path.isdir(in_dir):
        sys.exit(f"[error] chunk folder not found: {in_dir}")

    manifest_path = os.path.join(in_dir, "manifest.json")
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        ordered = [os.path.join(in_dir, c["file"]) for c in manifest["chunks"]]
        expected = {c["file"]: c["sha256"] for c in manifest["chunks"]}
        out_path = out_path or os.path.join(in_dir, manifest["name"])
    else:
        # manifest 없으면 폴더에서 .part* 를 긁어 이름순 정렬 (part000, part001, ... == 숫자순)
        ordered = sorted(glob.glob(os.path.join(in_dir, "*.part*")))
        expected = {}
        if not out_path:
            sys.exit("[error] no manifest.json; pass --out explicitly")

    if not ordered:
        sys.exit(f"[error] no chunk files found in {in_dir}")
    for p in ordered:
        if not os.path.isfile(p):
            sys.exit(f"[error] missing chunk: {p}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"joining {len(ordered)} chunk(s) -> {out_path}")

    full_hash = hashlib.sha256()
    with open(out_path, "wb") as fout:
        for i, part_path in enumerate(ordered):
            part_hash = hashlib.sha256()
            with open(part_path, "rb") as fin:
                while True:
                    block = fin.read(READ_BLOCK)
                    if not block:
                        break
                    fout.write(block)
                    full_hash.update(block)
                    part_hash.update(block)
            name = os.path.basename(part_path)
            if name in expected and part_hash.hexdigest() != expected[name]:
                sys.exit(f"[error] chunk sha256 mismatch (corrupt/incomplete upload): {name}")
            print(f"  [{i + 1}/{len(ordered)}] {name}  ok")

    if manifest is not None:
        if full_hash.hexdigest() != manifest["sha256"]:
            sys.exit("[error] FINAL sha256 mismatch -> reassembled file is corrupt")
        got = os.path.getsize(out_path)
        if got != manifest["size"]:
            sys.exit(f"[error] size mismatch: got {got}, expected {manifest['size']}")
        print(f"verified OK. sha256={full_hash.hexdigest()} size={human(manifest['size'])}")
    else:
        print(f"joined (no manifest, integrity NOT verified). sha256={full_hash.hexdigest()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="큰 파일을 청크로 분할")
    sp.add_argument("src", help="분할할 원본 파일 (예: dataset/after_0328_train.h5)")
    sp.add_argument("--out-dir", default="dataset/upload_chunks", help="청크 저장 폴더 (기본: upload_chunks)")
    sp.add_argument("--chunk-size", default="10G", help="청크당 크기 (기본: 10G)")

    jp = sub.add_parser("join", help="폴더의 청크를 재조립 + sha256 검증")
    jp.add_argument("in_dir", help="청크가 들어있는 폴더 (예: upload_chunks)")
    jp.add_argument("--out", default=None, help="복원할 파일 경로 (manifest 있으면 생략 가능)")

    args = ap.parse_args()
    if args.cmd == "split":
        split(args.src, args.out_dir, parse_size(args.chunk_size))
    elif args.cmd == "join":
        join(args.in_dir, args.out)


if __name__ == "__main__":
    main()
