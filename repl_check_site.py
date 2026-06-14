# %% [markdown]
# repl_check_site.py — consistency / integrity checks for an embedded site
# ============================================================================
# Cross-checks the 4 artifacts a finished site writes and flags any mismatch:
#   meta.json   (n_cells, patch_grid, embed_dim)
#   cells.parquet (one row/cell, cell_id 0..n-1, patch_ref "<site>/patches.zarr#i")
#   patches.zarr  (n_cells, gh, gw, C) + one written chunk per cell
#   cls_vecs.npy  (n_cells, C)
# A site is "OK" only if all counts agree, the grid matches, cell_id is contiguous,
# patch_ref indices line up, every zarr chunk is on disk, and sampled cells are non-empty/finite.

# %% CELL 1 — setup ----------------------------------------------------------------
import os
import sys
import json

os.environ.setdefault("DINO_MODEL", "dinov3_vitl16")
sys.path.insert(0, "src")

import numpy as np
import geopandas as gpd
import zarr

EMB_BASE = os.environ.get("DINO_EMB_BASE", "/mnt/ai/DeepThought/dino_embeddings")
EMB_ROOT = os.path.join(EMB_BASE, os.environ.get("DINO_MODEL", "dinov3_vitl16"))
print("EMB_ROOT:", EMB_ROOT)


def check_site(site_id, emb_root=EMB_ROOT, n_sample=8, verbose=True):
    """Return {'site_id','ok','n','problems':[...]} after cross-checking the 4 artifacts."""
    sd   = os.path.join(emb_root, site_id)
    part = os.path.join(emb_root, "cells", f"site_id={site_id}")
    zpath = os.path.join(sd, "patches.zarr")
    P = {"site_id": site_id, "ok": True, "n": None, "problems": []}
    def fail(m): P["ok"] = False; P["problems"].append(m)

    # --- existence ---
    need = {"cells.parquet": os.path.join(part, "cells.parquet"),
            "patches.zarr":  zpath,
            "cls_vecs.npy":  os.path.join(sd, "cls_vecs.npy"),
            "meta.json":     os.path.join(part, "meta.json")}
    for tag, p in need.items():
        if not os.path.exists(p):
            fail(f"missing {tag}")
    if not P["ok"]:
        return _report(P, verbose)

    meta = json.load(open(need["meta.json"]))
    man  = gpd.read_parquet(need["cells.parquet"])
    z    = zarr.open_array(zpath, mode="r")
    cls  = np.load(need["cls_vecs.npy"], mmap_mode="r")

    # --- counts agree across all 4 ---
    n_meta, n_par, n_zarr, n_cls = meta.get("n_cells"), len(man), int(z.shape[0]), int(cls.shape[0])
    P["n"] = n_zarr
    if len({n_meta, n_par, n_zarr, n_cls}) > 1:
        fail(f"n_cells mismatch: meta={n_meta} parquet={n_par} zarr={n_zarr} cls={n_cls}")

    # --- grid / embed dim ---
    gh, gw = meta.get("patch_grid", [None, None]); C = meta.get("embed_dim")
    if (int(z.shape[1]), int(z.shape[2]), int(z.shape[3])) != (gh, gw, C):
        fail(f"zarr grid {tuple(int(s) for s in z.shape[1:])} != meta ({gh},{gw},{C})")
    if int(cls.shape[1]) != C:
        fail(f"cls_vecs dim {int(cls.shape[1])} != embed_dim {C}")

    # --- cell_id contiguous 0..n-1, unique ---
    cid = np.sort(man["cell_id"].to_numpy())
    if not np.array_equal(cid, np.arange(n_par)):
        fail("cell_id not contiguous 0..n-1 / not unique")

    # --- patch_ref '#i' matches cell_id (order-safe link) ---
    ms = man.sort_values("cell_id")
    idx = ms["patch_ref"].str.rsplit("#", n=1).str[-1].astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(n_par)):
        fail("patch_ref '#index' does not match cell_id")

    # --- every zarr chunk actually on disk (one chunk/cell -> c/<i>/0/0/0) ---
    cdir = os.path.join(zpath, "c")
    n_chunk = len(os.listdir(cdir)) if os.path.isdir(cdir) else -1
    if n_chunk != n_zarr:
        fail(f"written zarr chunks {n_chunk} != n_cells {n_zarr} (interrupted write?)")

    # --- sample cells: finite + not all-zero (embeddings really present) ---
    rng = np.random.default_rng(0)
    bad = 0
    for i in rng.choice(n_zarr, min(n_sample, n_zarr), replace=False):
        a = np.asarray(z[int(i)]).astype(np.float32)
        if (not np.isfinite(a).all()) or float(np.abs(a).max()) == 0.0:
            bad += 1
    if bad:
        fail(f"{bad}/{min(n_sample, n_zarr)} sampled cells all-zero or non-finite")

    # --- soft cross-check vs ginfo (not fatal, just informative) ---
    gi = meta.get("ginfo", {})
    if gi.get("after_tile_filter") not in (None, n_zarr):
        P["problems"].append(f"(info) ginfo.after_tile_filter={gi.get('after_tile_filter')} != n_cells={n_zarr}")
    return _report(P, verbose)


def _report(P, verbose):
    if verbose:
        flag = "OK  " if P["ok"] else "FAIL"
        print(f"[{flag}] {P['site_id']}  n_cells={P['n']}")
        for m in P["problems"]:
            print(f"        - {m}")
    return P


# %% CELL 2 — check ONE site --------------------------------------------------------
check_site("BHP_Rehab_2024_MWER2024_10cm_v2_tytonai_rg")


# %% CELL 3 — check EVERY embedded site in the store (monitor a run) -----------------
import glob
done = sorted(os.path.basename(os.path.dirname(p)).replace("site_id=", "")
              for p in glob.glob(os.path.join(EMB_ROOT, "cells", "site_id=*", "cells.parquet")))
print(f"{len(done)} sites with cells.parquet in {EMB_ROOT}\n")
results = [check_site(s) for s in done]
bad = [r for r in results if not r["ok"]]
print(f"\n===== {len(results)-len(bad)}/{len(results)} OK | {len(bad)} with problems =====")
for r in bad:
    print(" ", r["site_id"], "->", "; ".join(r["problems"]))
