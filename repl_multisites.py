# %% [markdown]
# repl_multisites.py — multi-site driver (thin, over src/)
# =======================================================
# Edit SITES, dry-run the resolution (which sites have a webmap + tiles), then RUN.
# Resume-safe: re-running CELL 4 skips sites whose cells.parquet already exists.

# %% CELL 1 — path + imports --------------------------------------------------------
import os
import sys

# Single GPU: run everything on this physical device. MUST be set before any torch import
# (our modules import torch lazily, so setting it here is enough). Edit/remove as needed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("DINO_MODEL", "dinov3_vitl16")

# Inference dtype. "bf16" casts the model+input to bfloat16: weights ~half VRAM (7B: 25->12.5GB)
# and activations halved -> fixes OOM at high upscaling. Outputs are still stored fp32, and the
# upscaling (upsample/high_res) is untouched. Set to "fp32" to revert. MUST be set before import config.
os.environ.setdefault("DINO_DTYPE", "bf16")

try:                     # running as a file / in VS Code
    _ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:        # bare REPL paste (no __file__) -> cwd must be the repo root
    _ROOT = "."
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _ROOT)   # repo root -> import download_weights
sys.path.insert(0, _SRC)

# --- OUTPUT LOCATION (MUST be set before `import config`) ---------------------------
# Leave the next line ON to save patches.zarr + cells.parquet + webmaps to YOUR LOCAL DISK.
# Comment it out to write to the shared /mnt store instead (the real 127-site production run).
os.environ.setdefault("DINO_EMB_ROOT", os.path.join(_ROOT, "outputs", "embeddings_local"))

for _m in ("config", "transforms", "dino", "pca", "plots", "store", "pipeline"):
    sys.modules.pop(_m, None)   # reload fresh from src/ on re-run

import config            # noqa: E402  (sets the DINO env on import — must come first)
import pipeline          # noqa: E402

print("OUTPUT  ->", config.EMB_ROOT, "(local)" if config.EMB_ROOT.startswith(_ROOT) else "(/mnt shared)")
print("settings -> high_res:", config.HIGH_RES, "| dtype:", config.DINO_DTYPE, "| model:", config.DINO_MODEL)


# %% CELL 2 — choose the sites ------------------------------------------------------
# ALL = pipeline.all_site_keys()                 # every site in config/sites_to_resolutions.json
# print(f"{len(ALL)} sites in the catalog")
SITES = [
    'GeoNadir/UAV_Descoberta_Brazil',
    'BHP 2004/Manned Bens Oasis Nov20',
    'Caramulla 2020/Manned Caramulla Nov20',
    'Caramulla 2020/Manned Caramulla',
    'BHP Creeks 2021/Manned Caramulla Post Wet',
    'BHP Creeks 2021/Manned Fortescue River Post Dry_V01',
    'BHP Creeks 2021/Manned PowerSt Post Dry',
    'BHP Creeks 2021/Manned PowerSt Post Wet',
    'BHP Creeks 2021/Manned Yandicoogina Q3 October/Mindi',
    'BHP Creeks 2021/Manned Yandicoogina Q3 October/Yandicoogina',
    'BHP Creeks 2021/Manned Yandicoogina Q4 November/Mindy2nddeploy',
    'BHP Creeks 2021/Manned Yandicoogina Q4 November/WANNAMUNNA',
    'BHP Creeks 2022/Manned Caramulla Post Dry',
    'BHP Creeks 2022/Manned Fortescue River Post Wet',
    'BHP Creeks 2022/Manned PowerSt Post Dry',
    'BHP Creeks 2022/Manned Shovelana Post Dry',
    'BHP Creeks 2022/Manned Yandicoogina Q1 MarApr/Mindi',
    'BHP Creeks 2022/Manned Yandicoogina Q1 MarApr/Wannamunna',
    'BHP Creeks 2022/Manned Yandicoogina Q2 Jun/Mindy',
    'BHP Creeks 2022/Manned Yandicoogina Q2 Jun/Wannamunna',
    'BHP Creeks 2022/Manned Yandicoogina Q4 Nov/Mindy',
    'BHP Creeks 2022/Manned Yandicoogina Q4 Nov/Wannamunna',
    'BHP Phase 3/Manned Goldsworthy SO',
    'BHP Phase 3/Manned Nim Shayegap SO',
    'CM2020/Manned MAC',
    'EM2020/MWER',
    'CM2020/Manned Yandi 3band fixed',
    'BHP Phase 3/Manned Yarrie SO',
    'Telfer/2220 Mine Site Manned',
    'Telfer/2220 Access Road Manned',
    'Telfer/2220 Havieron Mine Site Manned',
    'Telfer/2220 Havieron Access Road Manned',
    'Telfer/2053 Havieron Analogue Road',
    'Telfer/2053 Mine Site Manned',
    'Telfer/2015 Analogue00 Manned',
    'Telfer/2015 Manned',
    'Telfer/1923 Manned',
    'BHP_Rehab_2025/BHP_2525 Rehab25-Chichester',
    'BHP_Rehab_2025/BHP_2525 Rehab25-Goldsworthy',
    'BHP_Rehab_2025/BHP_2525 Rehab25-Yandi',
    'BHP_Rehab_2025/BHP_2525 Rehab25-Yarrie-Nimingarra',
    'FMG/FMG_2511_Mindy and EHR_Weeds',
    'Telfer/GG_2517_Haveiron_EIA_RGBI_10cm_20210420',
    'Telfer/NEW_Havieron_Minesite_April2023_28351_20230429',
    '29Metals/29M_2451_GG_UAV',
]


# %% CELL 3 — dry-run: which sites resolve (webmap + tiles) vs skip — no embedding ---
ok, miss = [], []
for k in SITES:
    try:
        config.resolve_rgb(k)
        config.resolve_tiles(k)                   # checks the /mnt ObjectData tiles dir exists
        ok.append(k)
    except Exception as e:
        miss.append((k, str(e)[:70]))
print(f"resolvable: {len(ok)} | unresolved: {len(miss)}")
for k, e in miss:
    print("  skip", k, "|", e)


# %% CELL 3b — ensure DINO weights are downloaded PROPERLY (robust; skip if already complete) ---
# WITHOUT this, CELL 4 -> setup_activity falls back to the activity's own downloader (128 parallel
# range reads + full pre-allocation), which on a GCS timeout leaves a CORRUPT .pth that its
# existence-only check then serves forever. Our downloader is sequential, resumable, size-validated
# and returns instantly if the file is already complete.
import asyncio                  # noqa: E402
import dino                     # noqa: E402  (picks 7B vs ViT-L by this GPU's VRAM)
import download_weights as dw   # noqa: E402  (repo-root module)
_model = dino.pick_dino_model()                 # prints the GPU + chosen model
asyncio.run(dw.download(_model))                # verbose, resumable; no-op if already on disk


# %% CELL 4 — RUN (resume-safe; finished sites are skipped) --------------------------
res = pipeline.run_sites(ok, upsample=4)
print("embedded/ok:", len(res["done"]), "| skipped/failed:", len(res["fails"]))


# %% CELL 5 — (optional) 2-GPU on the server: sites split modulo across GPU 0 and 1 -----
# Option A — ONE command, one process per GPU (output interleaves):
# pipeline.run_all_gpus(gpus=(0, 1), site_keys=ok)
#
# Option B — TWO terminals (clean per-GPU progress bars), each its own GPU + shard:
#   CUDA_VISIBLE_DEVICES=0 python src/pipeline.py --all-sites --shard 0/2
#   CUDA_VISIBLE_DEVICES=1 python src/pipeline.py --all-sites --shard 1/2
# (or from the CLI in one go:  python src/pipeline.py --all-sites --gpus 0,1)
# Both are resume-safe and split sites[i::2], so the two GPUs never touch the same site.
