#!/usr/bin/env python
"""observe_tiles.py — peek at the patch embeddings a running job is producing.

Reads cells straight from a site's patches.zarr (no GPU, read-only, safe while training), reduces
each cell's (gh,gw,C) patch grid to a 3-channel PCA-RGB image, and saves a montage PNG. Only cells
actually written are shown (the zarr is pre-allocated to n_cells; unwritten cells read back as 0).

Two modes:
  - site_id  (no "/"):  PCA-RGB montage only.        e.g. BHP_Rehab_2024_MWER2024_10cm_v2_tytonai_rg
  - site_key (has "/"): source RGB | PCA-RGB pairs.  e.g. "BHP_Rehab_2024/MWER2024"
    (rebuilds the grid like the pipeline so it works mid-run, before cells.parquet exists)

Usage:
    python observe_tiles.py                         # newest site under EMB_ROOT, PCA only, 12 cells
    python observe_tiles.py <site_id> [n]           # PCA-only montage
    python observe_tiles.py "<Project>/<Site>" [n]  # RGB | PCA side-by-side
"""
import glob
import os
import sys

import numpy as np
import zarr

EMB_ROOT = os.environ.get("DINO_EMB_ROOT", "/mnt/ai/DeepThought/dino_embeddings")


def _newest_zarr():
    zarrs = glob.glob(os.path.join(EMB_ROOT, "*", "patches.zarr"))
    if not zarrs:
        sys.exit(f"no */patches.zarr under {EMB_ROOT}")
    return max(zarrs, key=os.path.getmtime)


def _written(zpath):
    """Indices of cells actually written = chunk dirs under patches.zarr/c/<i>/..., sorted."""
    cdir = os.path.join(zpath, "c")
    w = sorted(int(d) for d in os.listdir(cdir) if d.isdigit()) if os.path.isdir(cdir) else []
    if not w:
        sys.exit(f"no cells written yet in {zpath}")
    return w


def _pca_rgb(grids, sample=20000, seed=0):
    """Shared PCA(3) across all cells' patches -> per-cell (gh,gw,3) in [0,1], 2-98% stretch."""
    from sklearn.decomposition import PCA
    gh, gw, c = grids[0].shape
    flat = np.concatenate([g.reshape(-1, c) for g in grids], 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], min(sample, flat.shape[0]), replace=False)
    pca = PCA(n_components=3, random_state=seed).fit(flat[idx])
    proj = pca.transform(flat)
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    proj = np.clip((proj - lo) / (hi - lo + 1e-9), 0, 1)
    return [proj[i * gh * gw:(i + 1) * gh * gw].reshape(gh, gw, 3) for i in range(len(grids))]


def _grid_and_rgb(site_key):
    """Rebuild the cell grid (reprojected to the raster CRS) + rgb path, exactly as the pipeline
    does — so cell index i here == patches.zarr index i. Returns (site_id, grid_w, rgb_path)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    import config
    import transforms
    import rasterio
    res = config.default_res(site_key)
    site_id = config.site_id_from_key(site_key, res)
    site_dir = config.resolve_tiles(site_key, res)
    rgb_path = config.resolve_rgb(site_key, res)["rgb_path"]
    tiles = transforms.read_tile_bboxes(site_dir)
    tiles_clip, _ = transforms.crop_tiles_to_webmap(tiles, rgb_path)
    area = transforms.study_area(tiles_clip)
    grid, _ = transforms.build_tile_grid(area, tiles_clip, rgb_path, tile_patches=config.TILE_PATCHES,
                                         high_res=config.HIGH_RES, min_data_cov=config.MIN_DATA_COV)
    with rasterio.open(rgb_path) as r:
        wcrs = r.crs
    return site_id, grid.to_crs(wcrs), rgb_path


def _read_rgb(rgb_path, geom):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(rgb_path) as r:
        a = r.read((1, 2, 3), window=from_bounds(*geom.bounds, transform=r.transform),
                   boundless=True, fill_value=0)
    return a.transpose(1, 2, 0)                          # (H, W, 3) uint8


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rgb_mode = bool(arg) and "/" in arg

    if rgb_mode:
        site_id, grid_w, rgb_path = _grid_and_rgb(arg)
        zpath = os.path.join(EMB_ROOT, site_id, "patches.zarr")
        if not os.path.isdir(zpath):
            sys.exit(f"no patches.zarr at {zpath}")
    else:
        zpath = (os.path.join(EMB_ROOT, arg, "patches.zarr") if arg else _newest_zarr())
        if arg and not os.path.isdir(zpath):
            zpath = arg if os.path.basename(arg.rstrip("/")) == "patches.zarr" else sys.exit(f"no {zpath}")
        site_id = os.path.basename(os.path.dirname(zpath))

    z = zarr.open_array(zpath, mode="r")
    idxs = _written(zpath)[-n:]
    print(f"site: {site_id}\npatches.zarr {z.shape} {z.dtype} — showing {idxs[0]}..{idxs[-1]}"
          f"{'  (RGB | PCA)' if rgb_mode else '  (PCA only)'}")

    grids, rgbs = [], []
    for i in idxs:
        g = np.asarray(z[i]).astype(np.float32)
        grids.append(g)
        print(f"  cell_{i:04d}: {g.shape} NaN={np.isnan(g).any()} min={g.min():.2f} max={g.max():.2f}")
        if rgb_mode:
            rgbs.append(_read_rgb(rgb_path, grid_w.geometry.iloc[i]))
    imgs = _pca_rgb(grids)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rgb_mode:
        # one row per cell: [source RGB | PCA-RGB]
        fig, axes = plt.subplots(len(idxs), 2, figsize=(6, 3 * len(idxs)))
        axes = np.atleast_2d(axes)
        for row, (i, rgb, pca_img) in enumerate(zip(idxs, rgbs, imgs)):
            axes[row, 0].imshow(rgb); axes[row, 0].set_title(f"{i:04d} RGB", fontsize=8)
            axes[row, 1].imshow(pca_img); axes[row, 1].set_title(f"{i:04d} PCA", fontsize=8)
            for c in (0, 1):
                axes[row, c].axis("off")
        suffix = "_rgbpca"
    else:
        cols = min(4, len(imgs)); rows = (len(imgs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, img, i in zip(np.atleast_1d(axes).ravel(), imgs, idxs):
            ax.imshow(img); ax.set_title(f"{i:04d}", fontsize=8); ax.axis("off")
        suffix = ""
    fig.suptitle(f"{site_id} — {len(imgs)} cells", fontsize=10)
    fig.tight_layout()
    out = os.path.join("outputs", f"observe_{site_id}{suffix}.png")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
