# Drive-cycle speed profiles (third-party data)

These CSV files are **not authored by the repository authors**. They are speed-versus-time
profiles (1 Hz) taken, unmodified, from the NREL FASTSim repository
(https://github.com/NREL/fastsim, branch `fastsim-2`) at the commit recorded in
`SOURCE_COMMIT.txt` (`63808be8d09cb191047b536a2b9b7f883dbbd0c3`).

| File | Cycle | Use in this study |
|---|---|---|
| `udds.csv`, `hwfet.csv`, `NREL13.csv`, `TSDC_tripno_42648_cycle.csv` | UDDS, HWFET, NREL13, one TSDC trip | train |
| `wltc_low_3.csv`, `wltc_medium_3b.csv` | WLTC low / medium | validation |
| `us06.csv`, `wltc_high_3b.csv`, `wltc_extrahigh_3.csv` | US06, WLTC high, WLTC extra-high | test |
| `wmtc_all.csv`, `ftpmc1b.csv`, `longHaulDriveCycle.csv` | WMTC, FTP-MC1b, long-haul | additional held-out |

Only the cycles that the code actually reads are included (two further files present in the
working copy, `accel.csv` and `wltc_3b.csv`, are unused and were left out on purpose to keep
the amount of redistributed third-party data minimal).

## Licence status (read before redistributing)

* The FASTSim repository is published by NREL (now the National Laboratory of the Rockies)
  and its repository page lists the **Apache-2.0** licence. The repository page says nothing
  specific about licensing of the individual cycle files.
* The authors have **not verified file-by-file terms**. In particular, the name of `TSDC_tripno_42648_cycle.csv`
  indicates a trip from NREL's Transportation Secure Data Center, whose data carry their own
  terms of use (not checked), and some cycles are defined by regulatory bodies (US EPA, UNECE WLTP/WMTC).
* Our CC BY 4.0 / MIT grants do **not** apply to these files.
* If you need to redistribute them, fetch them from the FASTSim repository at the pinned
  commit and check the licence there. To use this repository without redistributing them,
  delete this folder and restore it from upstream.

If you use these cycles, please cite FASTSim (NREL) as requested in its repository.
