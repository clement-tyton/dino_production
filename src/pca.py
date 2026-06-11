"""GPU PCA (the one PCA used everywhere) + site PCA-RGB canvas/webmap.

Operates purely on the saved patch grids (one chunked Zarr per site, shape (gh,gw,C) per cell,
read via patch_io by ref) — NO model or activity import. torch is imported lazily inside
transform_all_tiles. 3 components -> the RGB web-map; 256 -> KMeans/BSP at scale.
"""
from __future__ import annotations

import os

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import from_bounds
from tqdm import tqdm

import config  # noqa: F401  (ensures env setup if pca is the first import)
import patch_io


def pca_rgb(emb, max_fit=200_000):
    """(C,H,W) embedding -> (H,W,3) in [0,1] from its top-3 PCA axes (2-98% stretch).

    Single-cell, in-memory (numpy). Used by plots.plot_cell before any .npz exists.
    """
    C, H, W = emb.shape
    X = emb.reshape(C, -1).T.astype(np.float32)
    Xc = X - X.mean(0)
    fit = Xc if len(Xc) <= max_fit else Xc[np.random.default_rng(0).choice(len(Xc), max_fit, replace=False)]
    _, V = np.linalg.eigh(fit.T @ fit)                 # covariance/eigh route (no giant SVD)
    proj = Xc @ V[:, ::-1][:, :3]
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    return np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1).reshape(H, W, 3)


class GPUPCA:
    """Fitted PCA (.mean_/.components_/.n_components_) for transform_all_tiles. Basis =
    covariance/eigh on a random patch subsample (the batch PCA). For a true all-patch fit,
    accumulate mean + X^T X over every cell, or use sklearn IncrementalPCA."""
    def __init__(self, refs, n_components=256, normalize=True, max_fit=500_000, seed=0):
        rng = np.random.default_rng(seed)
        per = max(1, max_fit // len(refs))
        fit = []
        for ref in tqdm(refs, desc="PCA fit", unit="tile"):
            a = patch_io.load(ref)
            a = a.reshape(-1, a.shape[-1]).astype(np.float32)
            if normalize:
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
            fit.append(a[rng.choice(len(a), min(len(a), per), replace=False)])
        X = np.concatenate(fit); self.mean_ = X.mean(0)
        _, V = np.linalg.eigh((X - self.mean_).T @ (X - self.mean_))
        self.components_ = np.ascontiguousarray(V[:, ::-1][:, :n_components].T)   # (n_components, C)
        self.n_components_ = n_components


def transform_all_tiles(refs, pca, *, normalize=True, show_progress=True, key_fn=None,
                        device="cuda", out_dtype=np.float16):
    """Project every cached tile into PCA space (one zarr read per cell; normalize + project on
    GPU; fp16 flat output with per-tile views). Returns (per_tile, flat, shapes, names)."""
    import torch
    key_fn = key_fn or patch_io.key
    shapes, names = [], []
    bar = tqdm(refs, desc="indexing shapes", unit="tile") if show_progress else refs
    for ref in bar:
        gh, gw = patch_io.shape(ref)
        shapes.append((gh, gw)); names.append(key_fn(ref))
    if len(set(names)) != len(names):
        dup = next(n for n in names if names.count(n) > 1)
        raise ValueError(f"Duplicate key from key_fn: {dup!r}. For cross-site use a namespaced key_fn.")
    total = sum(h * w for h, w in shapes)
    n_components = int(pca.n_components_)
    dev = torch.device(device)
    mean_t = torch.as_tensor(np.asarray(pca.mean_), dtype=torch.float32, device=dev)
    comp_t = torch.as_tensor(np.asarray(pca.components_), dtype=torch.float32, device=dev)
    per_tile_pca, flat_pca, off = {}, np.empty((total, n_components), dtype=out_dtype), 0
    it = (tqdm(zip(refs, names, shapes), total=len(refs), desc="PCA transform")
          if show_progress else zip(refs, names, shapes))
    for ref, name, (gh, gw) in it:
        pg = patch_io.load(ref); pg = pg.reshape(-1, pg.shape[-1])
        t = torch.from_numpy(np.ascontiguousarray(pg)).to(dev, dtype=torch.float32)
        if normalize:
            t = t / (t.norm(dim=1, keepdim=True) + 1e-6)
        z = (t - mean_t) @ comp_t.T
        n = z.shape[0]
        flat_pca[off:off + n] = z.cpu().numpy().astype(out_dtype, copy=False)
        per_tile_pca[name] = flat_pca[off:off + n].reshape(gh, gw, n_components)
        off += n
    return per_tile_pca, flat_pca, shapes, names


def site_pca_canvas(refs, geoms, pca=None, device="cuda"):
    """Project every patch with the GPU PCA (3 components), 2-98% stretch over all patches,
    paint each cell into a site array by its bbox. -> (canvas HxWx3 in [0,1], Affine, gsd)."""
    gh, gw, C = patch_io.full_shape(refs[0])
    b = [g.bounds for g in geoms]
    gsd = (b[0][2] - b[0][0]) / gw
    xmin, ymax = min(x[0] for x in b), max(x[3] for x in b)
    xmax, ymin = max(x[2] for x in b), min(x[1] for x in b)
    W, H = int(round((xmax - xmin) / gsd)), int(round((ymax - ymin) / gsd))
    pca = pca or GPUPCA(refs, n_components=3, max_fit=300_000)
    per_tile, flat, _, _ = transform_all_tiles(refs, pca, device=device,
                                               out_dtype=np.float32, show_progress=False)
    lo, hi = np.percentile(flat, 2, 0), np.percentile(flat, 98, 0)
    canvas = np.zeros((H, W, 3), np.float32)
    for ref, g in zip(refs, geoms):                       # paint each cell's projected patches
        tile = per_tile[patch_io.key(ref)]; ch, cw = tile.shape[:2]
        col, row = int(round((g.bounds[0] - xmin) / gsd)), int(round((ymax - g.bounds[3]) / gsd))
        canvas[row:row + ch, col:col + cw] = np.clip((tile - lo) / (hi - lo + 1e-6), 0, 1)
    return canvas, rasterio.Affine(gsd, 0, xmin, 0, -gsd, ymax), gsd


def webmap_from_manifest(site_id, emb_root=config.EMB_ROOT, webmap_path=None, out_tif=None):
    """Rebuild a site's PCA webmap from its saved cells.parquet (no re-embedding).

    Reads the manifest (cell_id, patch_ref, geometry) at <emb_root>/cells/site_id=<id>/, in
    cell_id order, and calls build_pca_webmap on the existing patches.zarr. Handy to re-render after a
    rendering change (e.g. the nodata fix). webmap_path enables the RGB nodata mask.
    """
    import geopandas as gpd
    man = gpd.read_parquet(os.path.join(emb_root, "cells", f"site_id={site_id}", "cells.parquet"))
    man = man.sort_values("cell_id")
    refs = [patch_io.absref(emb_root, r) for r in man["patch_ref"]]
    out_tif = out_tif or os.path.join(emb_root, site_id, "dino_pca_webmap.tif")
    return build_pca_webmap(refs, list(man.geometry), man.crs, out_tif, webmap_path=webmap_path)


def webmap_data_mask(webmap_path, transform, H, W):
    """(H, W) bool — True where the webmap has RGB data, over the canvas extent (one read).
    Used to mark nodata in the PCA webmap so real no-data areas are transparent in QGIS."""
    gsd = transform.a
    xmin, ymax = transform.c, transform.f
    xmax, ymin = xmin + W * gsd, ymax - H * gsd
    with rasterio.open(webmap_path) as r:
        m = r.read((1, 2, 3), window=from_bounds(xmin, ymin, xmax, ymax, transform=r.transform),
                   boundless=True, fill_value=0, out_shape=(3, H, W))
    return (m != 0).any(axis=0)


def build_pca_webmap(refs, geoms, crs, out_tif, webmap_path=None):
    """Render the site PCA-RGB canvas as a 4-band RGBA uint8 GeoTIFF (CRS + transform) -> QGIS.

    Transparency is carried by a real alpha band (band 4), which QGIS always honors -- unlike a
    per-band nodata value, which QGIS ignores for RGB color renderers (no-data then reads as
    black). alpha = where the webmap has RGB (if webmap_path is given) else where the canvas was
    painted. RGB is the raw 0..255 PCA colour (no 1..255 remap needed -- alpha, not 0, marks nodata).
    """
    canvas, transform, gsd = site_pca_canvas(refs, geoms)
    H, W = canvas.shape[:2]
    rgb = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    mask = (webmap_data_mask(webmap_path, transform, H, W) if webmap_path
            else ~(canvas == 0).all(axis=2))                    # fallback: unpainted background
    alpha = (mask.astype(np.uint8) * 255)                       # 255 = opaque data, 0 = transparent
    arr = np.concatenate([rgb, alpha[..., None]], axis=2).transpose(2, 0, 1)   # (4, H, W) band-major
    os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
    with rasterio.open(out_tif, "w", driver="GTiff", height=H, width=W, count=4,
                       dtype="uint8", crs=crs, transform=transform, photometric="RGB",
                       alpha="YES",                             # band 4 -> true alpha (ExtraSamples)
                       compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(arr)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    print(f"  {W}x{H}px @ {gsd:.2f} m | CRS {crs} | {int(mask.sum())}/{H*W} data px -> {out_tif}")
    return out_tif
