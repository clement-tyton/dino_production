# %% [markdown]
# repl_multisites.py — multi-site driver (thin, over src/)
# =======================================================
# Edit SITES, dry-run the resolution (which sites have a webmap + tiles), then RUN.
# Resume-safe: re-running CELL 4 skips sites whose cells.parquet already exists.

# %% CELL 1 — path + imports --------------------------------------------------------
import os
import sys

try:                     # running as a file / in VS Code
    _SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
except NameError:        # bare REPL paste (no __file__) -> cwd must be the repo root
    _SRC = "src"
sys.path.insert(0, _SRC)
for _m in ("config", "transforms", "dino", "pca", "plots", "store", "pipeline"):
    sys.modules.pop(_m, None)   # reload fresh from src/ on re-run

import config            # noqa: E402  (sets the DINO env on import — must come first)
import pipeline          # noqa: E402


# %% CELL 2 — choose the sites ------------------------------------------------------
ALL = pipeline.all_site_keys()                    # every site in config/sites_to_resolutions.json
print(f"{len(ALL)} sites in the catalog")
SITES = ALL                                       # or a hand-picked list, e.g.:
# SITES = ["BHP Creeks 2022/Manned Bens Oasis Post Dry", "BHP_Rehab_2024/MWER2024"]


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


# %% CELL 4 — RUN (resume-safe; finished sites are skipped) --------------------------
res = pipeline.run_sites(ok, make_webmap=True, make_plots=False)
print("embedded/ok:", len(res["done"]), "| skipped/failed:", len(res["fails"]))


# %% CELL 5 — (optional) 2-GPU: run this file in TWO terminals -----------------------
# Terminal A:  CUDA_VISIBLE_DEVICES=0 SHARD=0 python repl_multisites.py
# Terminal B:  CUDA_VISIBLE_DEVICES=1 SHARD=1 python repl_multisites.py
# shard = int(os.environ.get("SHARD", "0"))
# pipeline.run_sites(ok, shard=(shard, 2), make_webmap=True)
