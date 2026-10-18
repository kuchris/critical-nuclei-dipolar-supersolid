# Critical nuclei and nucleation barriers at the first-order superfluid–supersolid transition of a dipolar gas

Code: Zenodo *code archive: DOI to be added*.
Data: Zenodo *data archive: DOI to be added* (`data_C3a.zip`, 638 MB).

Code, run records and data of the paper

> W. K. Wong, *Critical nuclei and nucleation barriers at the first-order superfluid–supersolid transition of a dipolar gas*, arXiv (2026), *identifier to be added*.

The critical nuclei are saddle points of the grand potential of the
three-dimensional extended Gross–Pitaevskii equation for 164Dy in an infinite
tube at 6250 atoms per µm, computed with the GPU solver DipGPE included here
(`src/dipgpe`, PyTorch): a string method, Newton's method with MINRES at fixed
chemical potential or atom number, Hessian spectra by LOBPCG in parity sectors,
descents from the saddles and a staircase continuation of pinned states.

## Contents

| folder | content |
|---|---|
| `benchmarks/` | `tube_saddle.py` (string, Newton refinement, Hessian, asymmetric saddles, descents, staircase), `tube_coexistence.py` (Maxwell construction), `tube_localized.py` (localized states, interface tension), `tube_roton.py` (Bogoliubov roton spinodal), `c3a_figures.py` (Figures 1–8) |
| `runs/` | 113 run records: command, parameters, code version, environment, log and results (JSON: barriers, ΔN, line densities, Hessian eigenvalues, descent traces) of every computation |
| `docs/C3a_record.md` | log of the results and of the issues found and resolved |
| `src/dipgpe/`, `tests/` | the solver and its tests |
| `tools/export_c3a_data.py` | how the data archive was written (from the run records with their fields, which this repository leaves out) |

## Install

    uv sync            # Python 3.13, PyTorch with CUDA 12.8 (see pyproject.toml)
    uv run python -m pytest -q

## Data

The data archive holds the 22 critical nuclei of Table II as real fields in
double precision, with their records (see its README). Unpack it here, so that
`data_C3a/states/` exists. The fields of the other states (fixed-N saddles,
staircase states, minima; 12 GB) are not archived; their numbers are in the
run records.

## Reproduce the paper

| result | command |
|---|---|
| Figures 1–8 and the numbers they show (`result.json` of a new run record) | `uv run python benchmarks/c3a_figures.py --out figures` (Figs. 1a and 2 read two fields from `data_C3a`) |
| a critical nucleus at fixed μ | `uv run python benchmarks/tube_saddle.py refine --ensemble grand --dtypes c128 --a <a_s> --from-run <id>`, continued from a neighbouring saddle; the command of every computation is in `runs/<id>_*/meta.json` |
| coexistence window (Appendix D) | `uv run python benchmarks/tube_coexistence.py`, parameters in `runs/20261008-123945_tube_coexistence/meta.json` |
| localized states and σ (Appendix D) | `uv run python benchmarks/tube_localized.py`, parameters in `runs/20261008-130517_tube_localized/meta.json` |

A Newton refinement of one saddle at 32 cells takes about 30 s and 2 GiB, a
Hessian or asymmetric-saddle search about 2 min and 7.5 GiB, on one RTX 5070 Ti.

## Provenance

The run records name the git commit of the development repository in which
each computation was made; this repository is a snapshot of it restricted to
this paper (see `RELEASE_SOURCE.txt`). In each run record, `git.diff`
(uncommitted changes at run time) is restricted to the released code files.

## Citation

If you use this code or data, please cite the paper and the Zenodo archives
(identifiers to be added).

## License

MIT (see `LICENSE`).
