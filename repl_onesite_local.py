# %% [markdown]
# repl_onesite_local.py — LOCAL in / LOCAL out, ViT-L @ 2048 (high-res), one site
# ==============================================================================
# A thin local driver over pipeline.run_site. Everything is written to your LOCAL disk (not /mnt),
# the small ViT-L model is forced, and high-res is on -> 2048 input -> 128x128 patch grid at 10cm.
# Tiles are read from your local copy; the RGB webmap auto-resolves from /mnt (mounted) unless you
# set WEBMAP to a local raster. Run cell by cell.

# %% CELL 1 — env (MUST be set BEFORE `import config`) -------------------------------
import os
import sys

# --- LOCAL OUTPUT: all artifacts (patches/.npz, cells.parquet, meta.json, webmap, plots) go here ---
os.environ["DINO_EMB_ROOT"] = os.path.abspath("outputs/embeddings_vitl_hr")   # <- your local disk

# --- small model + high-res 2048 ---
# 10cm native -> activity picks patch 512 / upsample 2; high_res doubles upsample -> 4 -> 512*4 = 2048
# -> 2048/16 = 128x128 patch grid, ~0.39 m/patch.
os.environ["DINO_MODEL"]    = "dinov3_vitl16"   # force ViT-L (ignore the VRAM auto-pick)
os.environ["DINO_HIGH_RES"] = "1"               # 2048 input
os.environ["DINO_DTYPE"]    = "bf16"            # halve weight VRAM + bf16 activations (helps a 12GB GPU)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

try:                     # running as a file / in VS Code
    _ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:        # bare REPL paste -> cwd must be the repo root
    _ROOT = "."
sys.path.insert(0, os.path.join(_ROOT, "src"))
for _m in ("config", "transforms", "dino", "pca", "plots", "store", "pipeline"):
    sys.modules.pop(_m, None)               # reload fresh from src/ on re-run
import config            # noqa: E402  (reads the env above on import — must come first)
import pipeline          # noqa: E402

# LOCAL training tiles (copy under DINO_TRAIN_ROOT). The webmap is only on /mnt -> auto-resolved
# (mounted). To go FULLY local, set WEBMAP to a local .tif and pass rgb_path=WEBMAP in CELL 2.
TILES  = "/home/clement/local_copy_train_data/BHP Creeks 2022/Manned Bens Oasis Post Dry/10cm/v2_tytonai_rg"
WEBMAP = None            # None -> resolve_rgb() from /mnt; or "/path/to/local/RGB_webmap.tif"
print("OUT  :", config.EMB_ROOT)
print("model:", config.DINO_MODEL, "| high_res:", config.HIGH_RES, "| dtype:", config.DINO_DTYPE)
print("tiles:", TILES)


# %% CELL 2 — run the site (tiles local, output local, ViT-L @ 2048) -----------------
# resume=False forces a fresh run; flip to True to skip if this site already exists in OUT.
res = pipeline.run_site(site_dir=TILES, rgb_path=WEBMAP, out_root=config.EMB_ROOT,
                        resume=False)
print(f"\n[done] {res['site_id']} | {res.get('n_cells')} cells "
      f"| {res.get('ginfo')}\n  patches -> {res.get('site_dir')}/patches.zarr")


# %% CELL 3 — verify the run wrote the expected geometry (meta.json) ------------------
import json
meta = json.load(open(os.path.join(config.EMB_ROOT, res["site_id"], "meta.json")))
print(json.dumps({k: meta[k] for k in ("dino_model", "high_res", "upsample", "dino_dtype",
                                       "embed_dim", "patch_grid", "patch_ground_m", "n_cells")}, indent=2))
