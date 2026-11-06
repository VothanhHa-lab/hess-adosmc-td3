# Data licences

This repository mixes three kinds of data. They do **not** share one licence.

| Path | Origin | Licence / terms |
|---|---|---|
| `data/agents/` (trained TD3 actors, `.npz` weights and training metadata `.json`) | Produced by the authors with `scripts/train.py` | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) |
| `data/processed/`, `results/` (result JSON, tables, figures) | Produced by the authors with this code | CC BY 4.0 |
| `data/cycles/` (1 Hz speed profiles) | **Third party.** Copied unmodified from the NREL FASTSim repository, pinned at commit `63808be8d09cb191047b536a2b9b7f883dbbd0c3` (branch `fastsim-2`), see `data/cycles/SOURCE_COMMIT.txt` | **Not relicensed by us.** Each file keeps the terms of its source. See `data/cycles/README.md`. |

Because the cycle files are third-party, the CC BY 4.0 grant above applies only to
the first two rows. Do not read the MIT licence of the code, or CC BY 4.0, as
covering `data/cycles/`.

If you redistribute this repository, keep `data/cycles/README.md` and
`SOURCE_COMMIT.txt` with the cycle files.
