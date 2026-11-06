"""Run wrapper: builds inputs, calls the kernel, stores raw data + metadata."""
import hashlib
import json
import os
import platform
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from .design import design_mpc, dummy_mpc
from .kernel import LOG_COLS, simulate
from .params import CTRL_MPC, CTRL_NAMES, IDX, build_params


def code_hash():
    h = hashlib.sha256()
    here = os.path.dirname(__file__)
    for fn in sorted(os.listdir(here)):
        if fn.endswith(".py"):
            with open(os.path.join(here, fn), "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:12]


def run(cfg, ctrl, scen, seed=0, switched=False, overrides=None, log_every=1,
        noise=True, out_dir=None, tag=""):
    ov = dict(overrides or {})
    if not noise:
        ov.update({"sig_v": 0.0, "sig_i": 0.0})
    p = build_params(cfg, ctrl, switched=switched, overrides=ov)
    if ctrl == CTRL_MPC:
        mpc = design_mpc(p, int(cfg["control"]["mpc_N"]["value"]),
                         float(cfg["control"]["mpc_wmax"]["value"]))
    else:
        mpc = dummy_mpc()
    t0 = time.perf_counter()
    log, status, xf = simulate(p, scen["Tend"], int(log_every), int(seed), scen["P0"],
                               scen["seg_t"], scen["seg_dP"], scen["seg_tau"],
                               scen["rip_frac"], scen["rip_f"],
                               mpc["H"], mpc["Hinv"], mpc["F"], mpc["margin"])
    wall = time.perf_counter() - t0
    data = {c: log[:, i] for i, c in enumerate(LOG_COLS)}
    meta = {
        "run_id": uuid.uuid4().hex[:10],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "controller": CTRL_NAMES[ctrl], "ctrl_id": int(ctrl),
        "scenario": scen["name"], "scenario_source": scen["source"],
        "seed": int(seed), "switched": bool(switched), "noise": bool(noise),
        "tag": tag, "overrides": ov, "status": "ok" if status == 0 else "FAILED(bus out of range)",
        "Tend": scen["Tend"], "n_samples_logged": int(log.shape[0]), "log_every": int(log_every),
        "wall_time_s": wall, "code_hash": code_hash(),
        "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
        "params": {k: float(p[i]) for k, i in IDX.items()},
        "mpc_design": mpc["info"],
        "final_state": xf.tolist(),
    }
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        stem = f"{scen['name']}_{CTRL_NAMES[ctrl].replace(' ', '').replace('-', '')}_{tag}seed{seed}"
        np.savez_compressed(os.path.join(out_dir, stem + ".npz"),
                            **{k: v.astype(np.float32) if k not in ("t",) else v for k, v in data.items()})
        with open(os.path.join(out_dir, stem + ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        meta["file"] = stem
    return data, meta
