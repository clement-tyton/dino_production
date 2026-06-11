"""Patch-grid persistence — ONE chunked, compressed Zarr array per site (replaces per-cell .npz).

Layout: <emb_root>/<site_id>/patches.zarr  with shape (n_cells, gh, gw, C), chunk = 1 cell,
dtype float16, Blosc(zstd, clevel=5, bitshuffle). One file-tree per site instead of thousands of
.npz -> far fewer files on /mnt, partial per-cell reads (z[i]), fast writes (~0.05s codec/cell,
<< the GPU forward). fp16 is lossless vs our bf16 compute (10 mantissa bits > 7).

A *ref* identifies one cell: the string "<zarr_path>#<index>". Stored in the manifest relative to
emb_root; resolved back to an absolute ref for reading. All readers go through load()/shape()/key().
"""
from __future__ import annotations

import os

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

ZARR_NAME = "patches.zarr"


def _compressor():
    return BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle)


def zarr_path(site_dir: str) -> str:
    return os.path.join(site_dir, ZARR_NAME)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False                      # no such process
    except PermissionError:
        return True                       # exists but owned by another user -> alive
    except OSError:
        return False
    return True


def acquire_write_lock(site_dir: str) -> str:
    """Refuse to start a 2nd concurrent writer on a site — two processes calling create(overwrite)
    on the same patches.zarr clear each other's chunks + metadata and silently corrupt it (cells
    become 0). Writes <site_dir>/.writing.lock with our PID; steals a stale lock (dead PID)."""
    lock = os.path.join(site_dir, ".writing.lock")
    if os.path.exists(lock):
        try:
            old = int(open(lock).read().strip())
        except Exception:
            old = None
        if old and old != os.getpid() and _pid_alive(old):
            raise RuntimeError(
                f"{zarr_path(site_dir)} is already being written by PID {old} — refusing a second "
                f"writer (it would corrupt the zarr). Run ONE process per shard.")
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    return lock


def release_write_lock(lock: str) -> None:
    try:
        os.remove(lock)
    except OSError:
        pass


def create(site_dir: str, n_cells: int, gh: int, gw: int, c: int):
    """Create (overwrite) the site's patches.zarr and return the writable array. Chunk = 1 cell."""
    return zarr.create_array(
        store=zarr_path(site_dir), shape=(n_cells, gh, gw, c), chunks=(1, gh, gw, c),
        dtype="float16", compressors=[_compressor()], overwrite=True)


def make_ref(zpath: str, idx: int) -> str:
    return f"{zpath}#{idx}"


def _split(ref: str):
    p, i = ref.rsplit("#", 1)
    return p, int(i)


_cache: dict[str, "zarr.Array"] = {}


def _open(zpath: str):
    """Open a Zarr array read-only, cached per path (reads happen after the site is fully written)."""
    z = _cache.get(zpath)
    if z is None:
        z = zarr.open_array(zpath, mode="r")
        _cache[zpath] = z
    return z


def load(ref: str) -> np.ndarray:
    """One cell's patch grid -> (gh, gw, C) float16 (readers upcast to float32 as needed)."""
    p, i = _split(ref)
    return np.asarray(_open(p)[i])


def shape(ref: str) -> tuple[int, int]:
    """(gh, gw) for one cell, from Zarr metadata only — no array body read."""
    p, _ = _split(ref)
    s = _open(p).shape
    return int(s[1]), int(s[2])


def full_shape(ref: str) -> tuple[int, int, int]:
    """(gh, gw, C) for one cell, from Zarr metadata only."""
    p, _ = _split(ref)
    s = _open(p).shape
    return int(s[1]), int(s[2]), int(s[3])


def key(ref: str) -> str:
    """Stable per-cell name (e.g. 'cell_0007') for per-tile dict keys."""
    _, i = _split(ref)
    return f"cell_{i:04d}"


def relref(ref: str, emb_root: str) -> str:
    """Ref with its path made relative to emb_root — what goes in the manifest."""
    p, i = _split(ref)
    return f"{os.path.relpath(p, emb_root)}#{i}"


def absref(emb_root: str, stored: str) -> str:
    """Inverse of relref: rebuild an absolute, readable ref from a manifest entry."""
    rel, i = stored.rsplit("#", 1)
    return f"{os.path.join(emb_root, rel)}#{i}"
