"""Export the reduced data set of paper C3a for a public archive (Zenodo).

The 22 critical nuclei of Table II (fixed chemical potential, the barrier of
Fig. 3), as real fields in double precision, so that their energies, residuals
and Hessians can be recomputed on the grid of the paper:

  data_C3a/states/<a_s>_<run id>_<state>/psi.npz      psi(x, y, z), float64 (the states are real)
  data_C3a/states/<a_s>_<run id>_<state>/record.json  the saddle record of the run (barrier, Delta N, mu,
                                                      residuals, line density, Hessian where computed)
  data_C3a/states/<a_s>_<run id>_<state>/params.json  command, parameters, code version, environment
  data_C3a/index.csv                                  one row per state
  data_C3a/grid.json, README.md, EXPORT_SOURCE.txt, MANIFEST.sha256
  data_C3a.zip                                        the whole folder

Every other number of the paper (fixed-N saddles, staircase states, descents,
Hessian spectra, coexistence) is in the run records (JSON) of the code
repository; benchmarks/c3a_figures.py reads the two fields it draws from here when
the runs come without their data folders.

  uv run python tools/export_c3a_data.py
"""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import c3a_figures as F  # noqa: E402

OUT = ROOT / "data_C3a"
KEEP_META = ("id", "name", "params", "command", "git_commit", "git_dirty", "environment", "start", "end", "wall_s")
L_UM = F.UNITS.microns(1.0)

README = """# Data for "Critical nuclei and nucleation barriers at the first-order superfluid--supersolid transition of a dipolar gas" (W. K. Wong)

The 22 critical nuclei of Table II of the paper: saddle points of the grand
potential Omega = E - mu N of the three-dimensional extended Gross--Pitaevskii
equation for 164Dy in a periodic tube at the mean line density 6250 per um,
from the roton spinodal (89.86 a0) to the edge of coexistence (90.273 a0). Each
state is a Newton-converged stationary state at fixed chemical potential; its
energy above the uniform superfluid, Omega - Omega_U, is the barrier E_b.

## Files

| file | content |
|---|---|
| `index.csv` | one row per state: a_s, centring, tube length, barrier, Delta N, mu, N, residual (relative residual of the stationarity equation in the full space; for four older runs the last Newton residual), Hessian eigenvalues where computed in that run, run id, folder |
| `states/<a_s>_<run>_<state>/psi.npz` | `psi`: the field psi[x, y, z] (float64; the states are real) |
| `states/<a_s>_<run>_<state>/record.json` | the record of the computation (barrier, Delta N, mu, Newton residuals, line density, Hessian) |
| `states/<a_s>_<run>_<state>/params.json` | command, parameters, code version (git commit) and environment of the run |
| `grid.json` | grid, units and model parameters |
| `MANIFEST.sha256` | checksums; `EXPORT_SOURCE.txt`: the commit of the export |

## Units and grid

Lengths in l = sqrt(hbar / m omega_perp) = 0.641 um (omega_perp = 2 pi x 150 Hz),
energies in hbar omega_perp. |psi|^2 is the density in atoms per l^3, so that
sum(psi**2) dx dy dz = N (dx = dy = 0.25 l; dz = 2.76 um / 17 = 0.2533 l). The
transverse grid has 96 x 96 points over 24 l (index 48 is x = 0); along z the
tube holds 32 cells of 2.76 um (64 cells at 89.86-89.88 a0), periodic. Dipoles
point along y (the second axis). The line density in atoms per um is
psi**2 summed over x and y, times dx dy / (0.641 um per l).

    import numpy as np
    psi = np.load("states/90.000_20261018-035941_saddle/psi.npz")["psi"]   # (96, 96, 544)
    n_z = (psi**2).sum(axis=(0, 1)) * 0.25**2 / 0.6409970693791566        # atoms per um

The uniform superfluid at the same mu holds N_U = 552000 atoms in 32 cells; a
nucleus holds N_U + Delta N.

## Code

The code (DipGPE, the saddle tools in benchmarks/tube_saddle.py and the figure
script) and the run records of every computation in the paper are in the
repository of the paper (see its README). The other stationary states of the
paper (fixed-N saddles, staircase states, minima) are not included; their
numbers are in the run records there.
"""


def eigs(h):
    """Lowest Hessian eigenvalues of a record, as text (even/odd sectors or the full space)."""
    if not h:
        return ""
    if "even" in h:
        return "even " + " ".join(f"{v:+.4g}" for v in h["even"]["eigenvalues"]) + "; odd " + \
               " ".join(f"{v:+.4g}" for v in h["odd"]["eigenvalues"])
    return "full " + " ".join(f"{v:+.4g}" for v in h["eigenvalues"])


def state(a_s, ref):
    """(run id, state name, record, barrier, Delta N, mu, residual, Hessian) of a Table II entry."""
    if isinstance(ref, tuple):                                    # (staircase run, rung)
        rid, k = ref
        stair = F.load(rid, "tube_saddle_staircase.json")
        r = stair["rungs"][k - 1]
        rec = {"staircase": {key: v for key, v in stair.items() if key != "rungs"}, "rung": r}
        return rid, f"rung{k}", rec, r["omega_minus_uniform"], r["delta_N"], stair["mu"], r["full_residual"], \
            r.get("hessian")
    rec = F.record(ref)                                           # older records: the last Newton residual only
    resid = rec["full_residual"] if "full_residual" in rec else rec["residuals"][-1]
    return ref, "saddle", rec, F.barrier(ref), rec["delta_N"], rec["mu"], resid, rec.get("hessian")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "states").mkdir(parents=True)
    rows = []
    for a_s, ref, kind in F.BARRIERS:
        rid, name, rec, eb, dn, mu, resid, hess = state(a_s, ref)
        psi = F.field(rid, name)
        if psi.is_complex():
            assert float(psi.imag.abs().max()) == 0.0, f"{rid} {name}: not a real field"
            psi = psi.real
        psi = psi.numpy().astype(np.float64)
        meta = F.load(rid, "meta.json")
        cells = psi.shape[2] // F.NZ_CELL
        dz_l = F.CELL_UM / F.NZ_CELL / L_UM
        n_atoms = float((psi ** 2).sum() * F.DX * F.DX * dz_l)
        n_uniform = F.DENSITY * cells * F.CELL_UM
        assert abs(n_atoms - (n_uniform + dn)) < 1e-6 * n_uniform, f"{rid}: N = {n_atoms}, expected {n_uniform + dn}"
        folder = f"{a_s:.3f}_{rid}_{name}"
        dst = OUT / "states" / folder
        dst.mkdir()
        np.savez_compressed(dst / "psi.npz", psi=psi)
        (dst / "record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        (dst / "params.json").write_text(json.dumps({k: meta.get(k) for k in KEEP_META}, indent=1), encoding="utf-8")
        rows.append({"a_s": a_s, "centring": kind, "cells": cells, "nz": psi.shape[2],
                     "barrier_hbar_omega": f"{eb:.6f}", "delta_N": f"{dn:.3f}", "mu_hbar_omega": f"{mu:.12g}",
                     "N": f"{n_atoms:.3f}", "residual": f"{resid:.2g}", "hessian_eigenvalues": eigs(hess),
                     "run": rid, "state": name, "folder": f"states/{folder}"})
        print(f"{a_s:.3f} {kind:12s} {cells} cells  E_b {eb:.4f}  dN {dn:.1f}  {rid} {name}", flush=True)
    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    grid = {"atom": "164Dy", "a_dd_a0": 130.8, "omega_perp": "2 pi x 150 Hz", "l_um": L_UM,
            "energy_unit": "hbar omega_perp", "density_unit": "atoms per l^3 (|psi|^2)", "axes": "psi[x, y, z]",
            "dipoles": "along y", "nx": 96, "ny": 96, "dx_l": F.DX, "dy_l": F.DX, "L_perp_l": F.L_PERP,
            "x0_index": 48, "cell_um": F.CELL_UM, "nz_per_cell": F.NZ_CELL, "dz_um": F.CELL_UM / F.NZ_CELL,
            "dz_l": F.CELL_UM / F.NZ_CELL / L_UM, "boundary_z": "periodic",
            "mean_line_density_per_um": F.DENSITY, "dipolar_cutoff": "at half the transverse box (12 l)"}
    (OUT / "grid.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "EXPORT_SOURCE.txt").write_text(f"Written by tools/export_c3a_data.py at development commit {commit}.\n",
                                           encoding="utf-8")
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(OUT).as_posix()}"
             for f in sorted(OUT.rglob("*")) if f.is_file()]
    (OUT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    z = ROOT / "data_C3a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():                                       # the fields are compressed already
                zf.write(f, Path("data_C3a") / f.relative_to(OUT),
                         compress_type=zipfile.ZIP_STORED if f.suffix == ".npz" else zipfile.ZIP_DEFLATED)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{len(rows)} states; folder {size / 1e6:.0f} MB; {z.name} {z.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
