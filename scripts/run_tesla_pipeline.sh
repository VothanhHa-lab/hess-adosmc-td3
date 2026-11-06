#!/bin/bash
# Full re-run of the drive-cycle study with the Tesla Model 3 configuration (configs/nominal.yaml).
set -e
cd "$(dirname "$0")/.."
echo "[pipe] validation baselines $(date -u +%T)"
python scripts/validation_baselines.py
echo "[pipe] MPC-EMS horizon selection $(date -u +%T)"
python scripts/mpc_ems_select.py
python - <<'PY'
import json, re
sel = json.load(open("data/processed/mpc_ems_selection.json"))["selected_horizon_s"]
p = "configs/nominal.yaml"; s = open(p).read()
s = re.sub(r"mpc_ems: \{horizon_s: \{persistence: [0-9.]+, preview: [0-9.]+\}",
           "mpc_ems: {horizon_s: {persistence: %s, preview: %s}" % (sel["persistence"], sel["preview"]), s)
open(p, "w").write(s); print("[pipe] horizons", sel)
PY
echo "[pipe] train filter on/off $(date -u +%T)"
python scripts/train.py --filter both --seeds 11 22 33 --nproc 2
echo "[pipe] train filter v2 $(date -u +%T)"
python scripts/train.py --filter v2 --seeds 11 22 33 --nproc 2
echo "[pipe] DONE training $(date -u +%T)"
