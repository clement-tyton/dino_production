"""Persistence (store): per-site output dirs, the embed loop, and the manifest.

Named 'store' (not 'io') to avoid clashing with the stdlib ``io`` module on sys.path.

Output layout (per site):
  <emb_root>/<site_id>/patches.zarr                one chunked array (n_cells,gh,gw,C) fp16
  <emb_root>/<site_id>/{cls_vecs.npy, cells.fgb}  per-site CLS + QGIS view
  <emb_root>/cells/site_id=<site_id>/cells.parquet  Hive partition (geometry + cls + patch_ref + id)

DuckDB later, ALL sites at once (site_id comes from the partition path):
  duckdb.sql("SELECT site_id, cell_id, cls FROM
      read_parquet('<emb_root>/cells/**/*.parquet', hive_partitioning=true)")
"""
from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

import config  # noqa: F401
import patch_io
from dino import embed_cell_tokens


def site_emb_dirs(site_id, emb_root=config.EMB_ROOT):
    """Create + return (site_dir, part_dir) for a site under the embeddings root.

    site_dir holds patches.zarr + the per-site QGIS view; part_dir is the Hive cells partition.
    """
    site_dir = os.path.join(emb_root, site_id)
    part_dir = os.path.join(emb_root, "cells", f"site_id={site_id}")   # Hive partition
    os.makedirs(site_dir, exist_ok=True)
    os.makedirs(part_dir, exist_ok=True)
    return site_dir, part_dir


def embed_grid(act, model, device, grid_w, webmap_path, site_dir, upsample=None,
               show_bar=True, desc=""):
    """Embed every cell of grid_w into ONE chunked Zarr array (site_dir/patches.zarr). Keep ALL
    patches. fp16 + Blosc(zstd, bitshuffle): ~0.05s codec/cell << the GPU forward, ~1.4x smaller
    than raw, lossless vs our bf16 compute. The array is created on the first cell (once gh,gw,C
    are known) and written one cell per chunk; partial per-cell reads later via patch_io.load.

    Returns (refs, cls_vecs (n_cells, C)). refs[i] <-> patches.zarr index i <-> cls_vecs row i.
    """
    geoms = list(grid_w.geometry)
    zpath = patch_io.zarr_path(site_dir)
    lock = patch_io.acquire_write_lock(site_dir)        # refuse a 2nd concurrent writer (corruption guard)
    try:
        z, refs, cls_list = None, [], []
        it = tqdm(geoms, desc=desc[:22] or "cells", unit="cell") if show_bar else geoms
        for i, geom in enumerate(it):
            _, emb, cls, _ = embed_cell_tokens(act, model, device, tuple(geom.bounds),
                                               webmap_path, upsample=upsample)   # emb = (C, gh, gw)
            grid = emb.transpose(1, 2, 0).astype(np.float16)                 # (gh, gw, C)
            if z is None:
                z = patch_io.create(site_dir, len(geoms), *grid.shape)       # shape known now
            z[i] = grid
            refs.append(patch_io.make_ref(zpath, i))
            cls_list.append(cls)
        cls_vecs = np.asarray(cls_list, dtype=np.float32) if cls_list else np.zeros((0, 0), np.float32)
        return refs, cls_vecs
    finally:
        patch_io.release_write_lock(lock)


def write_manifest(grid_w, refs, cls_vecs, site_id, part_dir, emb_root=config.EMB_ROOT):
    """Stable cell_id linking geometry <-> patch ref (patches.zarr#i) <-> CLS (order-safe, CRS kept).

    Writes the Hive-partitioned GeoParquet (the source of truth), a FlatGeobuf for QGIS, and
    cls_vecs.npy. Hive partition dir encodes site_id, so DuckDB reads it from the path.
    """
    manifest = grid_w.reset_index(drop=True)[["geometry"]].copy()
    manifest.insert(0, "cell_id", np.arange(len(manifest)))               # cell_id == zarr index == cls row
    manifest["patch_ref"] = [patch_io.relref(r, emb_root) for r in refs]  # "<site_id>/patches.zarr#i"
    manifest["cls"] = cls_vecs.tolist()                                  # CLS vector inline (self-contained)
    manifest.to_parquet(os.path.join(part_dir, "cells.parquet"))         # Hive-partitioned GeoParquet
    manifest[["cell_id", "patch_ref", "geometry"]].to_file(
        os.path.join(emb_root, site_id, "cells.fgb"), driver="FlatGeobuf")  # QGIS view (reorders on read)
    np.save(os.path.join(emb_root, site_id, "cls_vecs.npy"), cls_vecs)    # cls_vecs[cell_id], order-stable
    return manifest


def write_site_meta(site_id, part_dir, ginfo, refs, *, dino_model, high_res, upsample,
                    emb_root=config.EMB_ROOT):
    """Make the embeddings SELF-DESCRIBING: a meta.json next to cells.parquet capturing the
    grid/upscale geometry, the model, and the git commit. Without this, the only record of the
    upsample / patch size / embedding dim is the patch_grid shape + cell bbox (derivable) or the
    stdout log (lost). patch_ground_m = cell_ground_m / gw. Written twice (Hive partition dir +
    per-site dir) so it's found from either side. Best-effort on commit; never fails the run."""
    import datetime
    import json
    import subprocess
    gh, gw, c = (None, None, None)
    if refs:
        gh, gw, c = patch_io.full_shape(refs[0])                          # (gh, gw, C) from zarr meta
    cell_m = ginfo.get("cell_ground_m")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
                                capture_output=True, text=True).stdout.strip() or None
    except Exception:
        commit = None
    meta = {
        "site_id": site_id,
        "dino_model": dino_model, "high_res": bool(high_res), "upsample": upsample,
        "dino_dtype": config.DINO_DTYPE,
        "embed_dim": c, "patch_grid": [gh, gw],
        "patch_px": ginfo.get("patch_px"), "native_res_m": ginfo.get("native_res_m"),
        "cell_ground_m": cell_m,
        "patch_ground_m": (cell_m / gw if (cell_m and gw) else None),
        "n_cells": ginfo.get("n_cells", len(refs)),
        "git_commit": commit,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "ginfo": ginfo,
    }
    for d in (part_dir, os.path.join(emb_root, site_id)):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return meta
