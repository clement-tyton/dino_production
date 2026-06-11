"""DINOv3 activity setup + per-cell embedding.

This is the ONLY module that touches the dinov3_embedding activity / torch model. All
activity/torch imports are LAZY inside the functions so that ``import config`` (which sets
the env vars) always runs first — the activity package is never imported at module load.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as tf_from_bounds

import config


@contextlib.contextmanager
def muted():
    """Silence the activity's per-cell prints."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        yield


def pick_dino_model(min_gb_7b=config.VRAM_GB_FOR_7B, verbose=True):
    """Choose dinov3_vit7b16 if the GPU has >= min_gb_7b of VRAM, else dinov3_vitl16.
    Prints the GPU + chosen model. Falls back to ViT-L if there's no CUDA device."""
    import torch
    if not torch.cuda.is_available():
        if verbose:
            print("GPU: none (CPU) -> dinov3_vitl16")
        return "dinov3_vitl16"
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1e9
    model = "dinov3_vit7b16" if gb >= min_gb_7b else "dinov3_vitl16"
    if verbose:
        print(f"GPU: {p.name} ({gb:.0f} GB VRAM) -> {model} "
              f"({'fits 7B' if model.endswith('7b16') else f'< {min_gb_7b:.0f} GB, using ViT-L'})")
    return model


def _trust_torch_load():
    """PyTorch >=2.6 defaults torch.load(weights_only=True); the DINOv3 7B checkpoint is in the
    legacy .tar format, which weights_only can't read -> load fails. The activity calls torch.load
    without weights_only, so we flip the default to False (safe: trusted company GCS weights, and
    the result only feeds model.load_state_dict). Idempotent — guarded so repeated per-site calls
    don't nest wrappers."""
    import torch
    if getattr(torch.load, "_trusted", False):
        return
    _orig = torch.load
    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)
    _load._trusted = True
    torch.load = _load


def setup_activity(webmap_path, grid_gdf, out_fgb=None,
                   dino_model=config.DINO_MODEL, high_res=config.HIGH_RES):
    """Instantiate the activity once + load its model. Returns (act, model, device, grid_in_raster_crs).

    The grid is reprojected to the raster CRS and written to a FlatGeobuf (the activity's bbox
    input). S3Mock(working_dir="/") roots the object store at "/" so absolute /mnt/... webmap
    paths resolve (a default mock would make the path cwd-relative -> 404).
    """
    import asyncio
    from tytonai.test.s3_mock import S3Mock
    from dinov3_embedding.io_schema.model import Input
    from dinov3_embedding.main import Dinov3Embedding

    if dino_model in (None, "auto"):
        dino_model = pick_dino_model()        # choose 7B vs ViT-L by GPU VRAM (+ print)
    out_fgb = out_fgb or os.path.join(config.PIC_DIR, "grid.fgb")
    with rasterio.open(webmap_path) as r:
        wcrs = r.crs
    grid_w = grid_gdf.to_crs(wcrs)                      # cell bounds must be in the raster CRS
    grid_w.to_file(out_fgb, driver="FlatGeobuf")
    inp = Input.model_validate({"bbox": out_fgb, "dino_model": dino_model, "high_res": high_res,
                                "rasters": [{"bands": ["RED", "GREEN", "BLUE"], "raster_file": webmap_path}]})
    act = Dinov3Embedding(inp, "", S3Mock(working_dir="/"))
    _trust_torch_load()                                # PyTorch 2.6 weights_only fix (legacy .tar 7B)
    model, device = _load_model_lowmem(act)            # CPU-load -> cast -> GPU (avoids the activity OOM)
    return act, model, device, grid_w


def _load_model_lowmem(act):
    """Memory-efficient replacement for act.load_model() — and where DINO_DTYPE is applied.

    The activity's own loader (dinov3_embedding/main.py:286-292) puts TWO fp32 copies on the GPU
    at once: it ``torch.load(map_location="cuda")`` the 26.8 GB state_dict AND builds the model on
    cuda -> ~53 GB peak -> OOM on a 48 GB card for the 7B (this is the "loading model to gpu" OOM).
    Worse, our earlier post-hoc ``model.to(bfloat16)`` ran AFTER that, so bf16 never helped the load.

    Here we load the state_dict into CPU RAM, build the model on CPU, cast to bf16 THERE (if
    requested), and only THEN move it to the GPU -> peak GPU = 13.4 GB (bf16) / 26.8 GB (fp32).
    Mirrors the activity's get_dino_model + load_state_dict exactly; the upsample/grid are untouched.
    """
    import asyncio
    import torch
    from dinov3_embedding.backbones import get_dino_model
    from dinov3_embedding.download import download_dino_model_async
    name = act.input_model.dino_model.value
    device = torch.device("cuda")
    local_file = asyncio.run(download_dino_model_async(name))     # already on disk -> returns path
    state_dict = torch.load(local_file, map_location="cpu")       # 26.8 GB in CPU RAM, NOT on the GPU
    model = get_dino_model(name, device=torch.device("cpu"))      # build the architecture on CPU
    model.load_state_dict(state_dict, strict=True)
    del state_dict                                                # free the CPU copy before the GPU move
    if config.DINO_DTYPE == "bf16":
        model = model.to(torch.bfloat16)                          # cast on CPU -> only 13.4 GB hits the GPU
        print("dtype: bfloat16 (weights ~half VRAM, activations halved; outputs cast back to fp32)")
    model = model.to(device)                                      # only now -> GPU
    model.eval()
    return model, device


def embed_cell(act, model, device, bbox, webmap_path):
    """Embed EXACTLY the bbox — read the cell verbatim (no activity padding/overlap), then
    upscale+embed with the activity's model. The RGB matches the raw webmap read pixel-for-pixel
    and the embedding covers ONLY this cell (no neighbour context). -> (rgb HWC, emb CHW, tf).

    (We bypass act.read_image_bands, which pads each box with a context margin and trims the
    embedding back; here there's nothing to trim because we never padded.)
    """
    with rasterio.open(webmap_path) as r:
        rgb = r.read((1, 2, 3), window=from_bounds(*bbox, transform=r.transform),
                     boundless=True, fill_value=0).transpose(1, 2, 0)
    emb = act.create_embedding(model, device, rgb)        # upscales by upsample, embeds (FP32)
    tf = tf_from_bounds(*bbox, emb.shape[2], emb.shape[1])  # embedding geotransform for this cell
    return rgb, emb, tf


def embed_cell_tokens(act, model, device, bbox, webmap_path, upsample=None):
    """Exact-cell read -> one forward -> (rgb, patch grid (C,gh,gw), cls (C,), transform).

    One forward via forward_features yields BOTH the patch tokens AND the CLS token.
    upsample: forward resize factor. None -> act.patch_upsample_factor (2 at 10cm). Pass 4 to
    test the 2048 upscaling (512*4) -> 128x128 patch grid, embed_gsd ~0.39 m, ~4x GPU mem.
    """
    import torch
    from PIL import Image
    from dinov3_embedding.main import make_transform
    up = act.patch_upsample_factor if upsample is None else upsample
    with rasterio.open(webmap_path) as r:
        rgb = r.read((1, 2, 3), window=from_bounds(*bbox, transform=r.transform),
                     boundless=True, fill_value=0).transpose(1, 2, 0)
    mdtype = next(model.parameters()).dtype         # fp32, or bf16 if DINO_DTYPE=bf16
    x = make_transform(rgb.shape[0] * up)(Image.fromarray(rgb)).unsqueeze(0).to(device=device, dtype=mdtype)
    with torch.inference_mode():
        f = model.forward_features(x)               # dict: x_norm_clstoken + x_norm_patchtokens
    cls = f["x_norm_clstoken"][0].float().cpu().numpy()             # (C,) -> fp32 for storage
    pt = f["x_norm_patchtokens"][0].float().cpu().numpy()           # (gh*gw, C) -> fp32 for storage
    g = int(round(pt.shape[0] ** 0.5))
    patch = pt.reshape(g, g, -1).transpose(2, 0, 1)                 # (C, gh, gw)
    return rgb, patch, cls, tf_from_bounds(*bbox, patch.shape[2], patch.shape[1])
