"""Critical nucleus of the tube crystal at fixed N (paper C3, PLAN C3-2): string
method (dipgpe.mep) from the metastable uniform superfluid to the crystal.

Endpoints at mean density n (default 6250/um) in a periodic tube of ``cells``
crystal cells: the uniform state and the crystal, from single-cell ground
states (complex128, tube_localized.cell_state) tiled along z (the tiled cell
is an exact stationary state with the same energy per atom). The initial
string passes through a seed with a crystal patch of ``patch`` cells centred
at z = 0 in the uniform background. Every image is even in z, which pins the
translation of the nucleus (the string dynamics keep the parity).

Modes
  cost   peak GPU memory and time per string iteration for each (images,
         dtype): a few iterations each, no convergence.
  pilot  the string at one a_s within a time budget: convergence history
         (normal gradient, climbing residual, image energies), the line
         densities of the last string, and the string itself (data/).
  summary table and figure of refine runs (``--runs``): barrier, Morse index,
         nucleus length (full width at half maximum of the one-cell contrast),
         peak contrast, droplets and background depletion; no GPU.
  asym   the symmetry-broken saddle next to a symmetric one of index 2 (``--from-run``
         a refine run with its Hessian modes): dipgpe.mep.break_symmetry along the
         odd unstable mode, with the translation fixed by a phase condition
         (d psi/dz of the symmetric state), the amplitudes ``--amplitudes`` in units
         of the nucleus norm ||psi - psi_uniform||; then the lowest
         ``--index-modes`` Hessian eigenvalues in the full space (no parity
         sectors, translation deflated). Reviews: docs/reports/C3e_*.
  connect connectivity of a saddle at fixed mu (``--from-run`` a refine or asym run
         with its Hessian modes): dipgpe.mep.descend from the saddle +- eps times
         its unstable mode (eps = ``--eps`` x nucleus norm). The shrinking side
         should return to the uniform state; the growing side should lengthen the
         nucleus while Omega falls below the saddle (stopped once the nucleus is
         ``--grow-cells`` cells longer). Symmetric saddles descend in the even
         sector, asym ones with a translation phase condition.
  staircase the stationary states along the growth path at fixed mu, one by one
         (``--from-run`` a stationary state, ``--prev-run``/``--prev-state`` the one
         before it on the path): from each state, dipgpe.mep.break_symmetry along
         the right-front part (z > 0) of the last step finds the next stationary
         state (a saddle after a minimum, a minimum after a saddle; amplitudes
         ``--amplitudes`` in units of the last step), with its index in the full
         space; ``--rungs`` states. One front advances at a time.
  hessian lowest Hessian eigenvalues of a refined saddle (``--from-run`` a refine
         run) in the even and odd sectors of z -> -z (translation deflated in the odd
         one), with the parity of each mode.
  refine Newton refinement (dipgpe.mep.refine_saddle) in the given precision,
         from an image of a stored string or from the saddle of another refine
         run (continuation in a_s; a shorter tube is embedded in the uniform
         state of a longer one); the barrier against the uniform state (tiled
         cell, same precision) and, with --hessian, the Morse index by parity
         sectors (--sector none: no symmetry imposed, the translation fixed by a
         phase condition, the index in the full space). --ensemble grand fixes mu at that of the uniform state
         instead of N: the barrier is then Omega = E - mu N of the saddle minus
         that of the uniform state, the L -> infinity limit without the
         depletion term of the fixed-N tube (ISSUES C3-13), and delta_N the
         atoms the nucleus adds (negative: expelled). --grow m starts from the
         nucleus lengthened by m cells on each side (staircase seeds).

Lengths in l, energies in hbar omega_perp; densities in atoms per micron.

Usage:
  uv run python benchmarks/tube_saddle.py cost --a 90.0 --images 9,17 --dtypes c64,c128
  uv run python benchmarks/tube_saddle.py pilot --a 90.0 --images 9 --minutes 10
  uv run python benchmarks/tube_saddle.py refine --a 90.0 --from-run 20261016-173115 --dtypes c128
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
from dipgpe.bdg import constrained_hessian  # noqa: E402
from dipgpe.mep import FieldSpace, break_symmetry, descend, path_through, refine_saddle, string_method  # noqa: E402
from tube_coexistence import PER_L  # noqa: E402
from tube_localized import cell_state  # noqa: E402
from tube_supersolid import UNITS, tube_model  # noqa: E402

DTYPES = {"c64": torch.complex64, "c128": torch.complex128}


def endpoints(args):
    """Cell length, transverse points and the single-cell uniform and crystal states."""
    L = args.cell_um / UNITS.microns(1.0)
    Np = int(round(args.L_perp / args.dx))
    cg = dipgpe.Grid((Np, Np, args.nz_cell), (args.L_perp, args.L_perp, L), dtype=torch.complex128, device="cuda")
    psi_U, s_U = cell_state(args.a, cg, args.density, False)
    psi_M, s_M = cell_state(args.a, cg, args.density, True)
    return L, Np, psi_U, psi_M, {"uniform_cell": s_U, "crystal_cell": s_M}


def tube(args, L, Np, dtype):
    grid = dipgpe.Grid((Np, Np, args.cells * args.nz_cell), (args.L_perp, args.L_perp, args.cells * L),
                       dtype=dtype, device="cuda")
    return grid, tube_model(grid, args.a)


def initial_string(space, args, psi_U, psi_M, n_images):
    """Uniform -> crystal patch of ``args.patch`` cells (even about z = 0) -> crystal.

    ``--phase gap``: the tiled cells put droplets at half-integer cell positions, so
    a gap lies at z = 0. ``--phase droplet``: the crystal is translated by half a
    cell (spectrally; 17 points per cell) so that a droplet sits at z = 0, and the
    patch is a smooth window (width ``patch`` cells, edges half a cell) in the
    uniform state."""
    grid = space.grid
    dtype = grid.dtype
    c, K = args.cells, args.patch
    U = psi_U.repeat(1, 1, c).to(dtype)
    M = psi_M.repeat(1, 1, c).to(dtype)
    if args.phase == "gap":
        m0 = c // 2 - K // 2
        seed = torch.cat([psi_U.repeat(1, 1, m0), psi_M.repeat(1, 1, K), psi_U.repeat(1, 1, c - m0 - K)],
                         dim=2).to(dtype)
    else:
        L = args.cell_um / UNITS.microns(1.0)
        M = dipgpe.translate(grid, M, (0.0, 0.0, L / 2)).real.to(dtype)
        z = grid.x64[2]
        w = (0.5 * (torch.tanh((z + K * L / 2) / (L / 2)) - torch.tanh((z - K * L / 2) / (L / 2)))).to(grid.real_dtype)
        seed = U * (1 - w) + M * w
    return path_through(space, [U, seed, M], n_images)


def cost(args, run):
    L, Np, psi_U, psi_M, cells = endpoints(args)
    rows = []
    k = args.iters
    for name in args.dtypes.split(","):
        for n in [int(v) for v in args.images.split(",")]:
            row = {"dtype": name, "images": n, "grid": [Np, Np, args.cells * args.nz_cell]}
            try:
                torch.cuda.empty_cache()
                grid, model = tube(args, L, Np, DTYPES[name])
                space = FieldSpace(model)
                X0 = initial_string(space, args, psi_U, psi_M, n)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = {}

                def mark(it, X, info):
                    torch.cuda.synchronize()
                    times[it] = time.time()
                string_method(space, X0, step=args.step, max_iter=k, strict=False, climb=False, callback=mark,
                              callback_every=1)
                row["ms_per_iteration"] = 1e3 * (times[k] - times[2]) / (k - 2)
                row["peak_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
                row["field_MiB"] = grid.size * torch.empty((), dtype=grid.dtype).element_size() / 2 ** 20
                del X0, space, model, grid
            except torch.cuda.OutOfMemoryError:
                row["oom"] = True
            torch.cuda.empty_cache()
            rows.append(row)
            print(row, flush=True)
    run.save_json("tube_saddle_cost.json", {"cells": cells, "rows": rows})
    run.result(rows=rows)


def pilot(args, run):
    L, Np, psi_U, psi_M, cells = endpoints(args)
    grid, model = tube(args, L, Np, DTYPES[args.dtypes])
    space = FieldSpace(model)
    n = int(args.images)
    if args.from_run:                                  # shorter string: images 0..end_image of a stored string
        _, stored = _stored(args.from_run)
        end = args.end_image
        if end == "auto":                              # first image past the highest interior one
            import json
            folder = next(dipgpe.runs.runs_dir().glob(args.from_run + "*"))
            E = json.loads((folder / "tube_saddle_pilot.json").read_text(encoding="utf-8"))["energies_minus_E0"]
            end = min(max(range(1, len(E) - 1), key=lambda i: E[i]) + 1, len(E) - 1)
        X0 = path_through(space, [f.to("cuda", grid.dtype) for f in stored[:int(end) + 1]], n)
    else:
        X0 = initial_string(space, args, psi_U, psi_M, n)
    history = []
    t0 = time.time()

    def log(it, X, info):
        E = info["energies"].to(torch.float64).cpu()
        rel = (E - E[0]).tolist()
        history.append({"iteration": it, "seconds": time.time() - t0, "perpendicular": info["perpendicular"],
                        "residual": info["residual"], "saddle": info["saddle"], "E_minus_E0": rel})
        print(f"  it {it:6d}  {time.time() - t0:7.1f} s  normal {info['perpendicular']:.3e}  "
              f"saddle {info['saddle']}  residual {info['residual']}  max(E - E0) {max(rel):.4f}", flush=True)
        return time.time() - t0 > 60 * args.minutes
    X, info = string_method(space, X0, step=args.step, max_iter=10 ** 9, strict=False, callback=log,
                            callback_every=args.every, climb_tol=args.climb_tol,
                            sector=lambda f: 0.5 * (f + mirror(f)))
    line = ((X.abs() ** 2).sum((1, 2)) * grid.dx[0] * grid.dx[1]).double().cpu()
    E = info["energies"]
    torch.save(X.cpu(), run.data("string.pt"))
    run.save_json("tube_saddle_pilot.json", {
        "cells": cells, "N": float(args.density * PER_L * args.cells * L), "history": history,
        "energies_minus_E0": (E - E[0]).tolist(), "saddle": info["saddle"], "iterations": info["iterations"],
        "converged": info["converged"], "line_density_per_um": (line / PER_L).tolist(),
        "dz_um": UNITS.microns(grid.dx[2])})
    run.result(iterations=info["iterations"], seconds=time.time() - t0, converged=info["converged"],
               perpendicular=info["perpendicular"], residual=info["residual"], saddle=info["saddle"],
               max_E_minus_E0=float((E - E[0]).max()))


def _stored(run_id):
    folder = next(dipgpe.runs.runs_dir().glob(run_id + "*"))
    return folder, torch.load(folder / "data" / "string.pt")


def mirror(f):
    """f(-z) on the periodic grid (x_0 = -L/2)."""
    return torch.roll(torch.flip(f, [-1]), 1, dims=-1)


def start_state(args, grid, psi_U):
    """Initial state for Newton from ``--from-run``: the saddle of a refine run (any a_s,
    continuation) or an image of a pilot string; a shorter tube is embedded in the
    middle of the uniform state of this a_s (same transverse grid and cell)."""
    folder = next(dipgpe.runs.runs_dir().glob(args.from_run + "*"))
    if (folder / "data" / f"{args.state}.pt").exists():
        psi0, image = torch.load(folder / "data" / f"{args.state}.pt"), None
    elif args.state != "saddle":
        raise SystemExit(f"{folder.name} has no data/{args.state}.pt")
    else:
        image = args.image
        if image is None:
            import json
            pj = json.loads((folder / "tube_saddle_pilot.json").read_text(encoding="utf-8"))
            image = pj["saddle"]
            if image is None:                          # no climbing image: the highest interior image
                E = pj["energies_minus_E0"]
                image = max(range(1, len(E) - 1), key=lambda i: E[i])
        psi0 = torch.load(folder / "data" / "string.pt")[image]
    psi0 = psi0.to("cuda", torch.complex128)
    import json
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    src = meta.get("params") or meta.get("parameters")
    if (src["L_perp"], src["dx"], src["nz_cell"]) != (args.L_perp, args.dx, args.nz_cell):
        # another grid (C3-2 (f)): a larger transverse box is padded with zeros at the
        # source spacing, then the field is Fourier-interpolated to this spacing
        L = args.cell_um / UNITS.microns(1.0)
        Np_s = psi0.shape[0]
        if src["L_perp"] != args.L_perp:
            add = int(round((args.L_perp - src["L_perp"]) / src["dx"]))
            if add < 0 or add % 2:
                raise SystemExit(f"cannot pad L_perp {src['L_perp']} to {args.L_perp} at dx {src['dx']}")
            psi0 = torch.nn.functional.pad(psi0, (0, 0, add // 2, add // 2, add // 2, add // 2))
            Np_s += add
        g_from = dipgpe.Grid((Np_s, Np_s, psi0.shape[2]), (args.L_perp, args.L_perp, psi0.shape[2] // src["nz_cell"] * L),
                             dtype=torch.complex128, device="cuda")
        g_to = dipgpe.Grid((grid.shape[0], grid.shape[1], psi0.shape[2] // src["nz_cell"] * args.nz_cell),
                           g_from.length, dtype=torch.complex128, device="cuda")
        psi0 = dipgpe.resample(psi0, g_from, g_to).real.to(torch.complex128)
    psi0 = psi0.to(grid.dtype)
    extra = grid.shape[2] - psi0.shape[2]
    if extra:
        pad = extra // (2 * args.nz_cell)
        if extra < 0 or extra % (2 * args.nz_cell) or psi0.shape[:2] != grid.shape[:2]:
            raise SystemExit(f"cannot embed a tube of {psi0.shape} into {grid.shape}")
        side = psi_U.repeat(1, 1, pad).to(grid.dtype)
        psi0 = torch.cat([side, psi0, side], dim=2)
    if args.grow:
        psi0 = grow(psi0, args.nz_cell, args.grow)
    if args.ensemble == "grand":                     # keep the start's norm: renormalizing to N moved a stored
        return psi0, folder.name, image              # fixed-mu state off its stationary point
    return dipgpe.normalize(grid, psi0, float(dipgpe.norm(grid, psi_U.repeat(1, 1, args.cells).to(grid.dtype)))), \
        folder.name, image


def grow(psi, nz_cell, m):
    """A longer nucleus as a Newton seed (C3-2 (e) staircase): the period centred at
    z = 0 is repeated 2m + 1 times and each front moves out by m cells, taken from the
    uniform background at the tube ends; the centre (droplet or gap) is kept and the
    nucleus gains 2m droplets."""
    nz, c, h, s = psi.shape[-1], psi.shape[-1] // 2, nz_cell // 2, m * nz_cell
    if nz_cell % 2 == 0 or c - h - s < nz_cell:
        raise SystemExit(f"cannot grow the nucleus by {m} cells on each side")
    core = psi[..., c - h:c + h + 1]
    return torch.cat([psi[..., s:c - h], core.repeat(1, 1, 2 * m + 1), psi[..., c + h + 1:nz - s]], dim=-1)


def _embed_z(f, nz):
    """Fields (leading axes kept) zero-padded symmetrically along z to nz points."""
    extra = nz - f.shape[-1]
    if extra == 0:
        return f
    pad = torch.zeros(*f.shape[:-1], extra // 2, dtype=f.dtype, device=f.device)
    return torch.cat([pad, f, pad], dim=-1)


def stored_modes(folder, grid):
    """Hessian modes saved by an earlier refine or hessian run, embedded in this tube."""
    f = folder / "data" / "hessian_modes.pt"
    if not f.exists():
        return None
    modes = torch.load(f)
    return {k: _embed_z(v.to("cuda", torch.complex128), grid.shape[2]) for k, v in modes.items()}


def parity_index(model, grid, psi, tol, iters, x0=None, save=None, mu=None):
    """Lowest Hessian eigenvalues in the even (two) and odd (one, translation deflated)
    sectors of z -> -z, with the parity of each mode; ``x0`` (dict per sector) warm-starts
    from the modes of a nearby saddle, ``save`` (path) stores this run's modes; ``mu``
    (grand canonical) the Hessian of E - mu N without the norm constraint."""
    dz = grid.ifft(1j * grid.k[2] * grid.fft(psi))
    rows, keep = {}, {}
    for name, sign, n in (("even", 1.0, 2), ("odd", -1.0, 1)):
        t0 = time.time()
        sector = lambda f, s=sign: 0.5 * (f + s * mirror(f))      # noqa: E731
        lam, modes, info = constrained_hessian(model, psi, n_modes=n, deflate=[dz] if sign < 0 else None,
                                               tol=tol, max_iter=iters, sector=sector, mu=mu,
                                               x0=None if x0 is None else x0.get(name))
        keep[name] = modes.to(torch.complex64).cpu()
        parity = [float((m - mirror(m)).abs().norm() / (m + mirror(m)).abs().norm().clamp_min(1e-300)) for m in modes]
        rows[name] = {"eigenvalues": lam.tolist(), "residuals": info["residual"], "iterations": info["iterations"],
                      "odd_over_even_norm": parity, "seconds": time.time() - t0}
        print(f"  Hessian {name}: {lam.tolist()} residuals {info['residual']} "
              f"({info['iterations']} it, {time.time() - t0:.0f} s)", flush=True)
    rows["index"] = sum(v < 0 for v in rows["even"]["eigenvalues"]) + sum(v < 0 for v in rows["odd"]["eigenvalues"])
    rows["certified"] = all(r < 1 for k in ("even", "odd") for r in rows[k]["residuals"])
    if save is not None:
        torch.save(keep, save)
    return rows


def refine(args, run):
    L, Np, psi_U, psi_M, cells = endpoints(args)
    grid, model = tube(args, L, Np, DTYPES[args.dtypes])
    psi0, source, image = start_state(args, grid, psi_U)
    warm = (stored_modes(next(dipgpe.runs.runs_dir().glob((args.modes_from or args.from_run) + "*")), grid)
            if args.hessian else None)
    N = float(args.density * PER_L * args.cells * L)
    E_U, mu = _uniform(model, grid, psi_U, args.cells, N, args.ensemble)
    t0 = time.time()
    steps = []

    def log(k, psi, res):
        steps.append({"step": k, "seconds": time.time() - t0, "residual": res})
        print(f"  Newton {k}: residual {res:.3e}  ({time.time() - t0:.0f} s)", flush=True)
    torch.cuda.reset_peak_memory_stats()
    even = lambda f: 0.5 * (f + mirror(f))                       # noqa: E731
    odd_start = float((psi0 - mirror(psi0)).abs().norm() / psi0.abs().norm())
    phase = args.sector == "none"                    # translation fixed by a phase condition instead
    psi, info = refine_saddle(model, psi0, mu=mu, max_newton=args.max_newton, strict=False, callback=log,
                              sector=None if phase else even,
                              deflate=[_dz(grid, psi0).real.to(grid.dtype)] if phase else None)
    dN = info["N"] - N
    barrier = info["energy"] - E_U - (0.0 if mu is None else mu * dN)
    print(f"barrier {'E' if mu is None else 'Omega'} - uniform = {barrier:.6f} hbar omega, "
          f"delta_N = {dN:.2f}, residual {info['residual']:.2e}, "
          f"MINRES {info['linear_iterations']}{', stalled' if info['stalled'] else ''}, "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    line = ((psi.abs() ** 2).sum((0, 1)) * grid.dx[0] * grid.dx[1]).double().cpu()
    torch.save(psi.cpu(), run.data("saddle.pt"))
    out = {"cells": cells, "N": N, "ensemble": args.ensemble, "mu": mu, "delta_N": dN, "from_run": source,
           "image": image, "E_uniform": E_U,
           "odd_part_of_start": odd_start, "odd_part": float((psi - mirror(psi)).abs().norm() / psi.abs().norm()),
           "E_saddle": info["energy"], "barrier": barrier, "residuals": info["residuals"],
           "linear_iterations": info["linear_iterations"], "steps": steps, "converged": info["converged"],
           "stalled": info["stalled"], "newton_seconds": time.time() - t0,
           "full_residual": info["full_residual"], "multipliers": info["multipliers"],
           "newton_peak_GiB": torch.cuda.max_memory_allocated() / 2 ** 30,
           "line_density_per_um": (line / PER_L).tolist(), "dz_um": UNITS.microns(grid.dx[2])}
    run.save_json("tube_saddle_refine.json", out)
    run.result(barrier=barrier, delta_N=dN, residual=info["residual"], converged=info["converged"],
               stalled=info["stalled"], newton_steps=info["iterations"], minres=info["linear_iterations"],
               seconds=time.time() - t0)
    if args.hessian:
        del psi0
        torch.cuda.empty_cache()                       # Newton's cached blocks fragment the Hessian's memory
        torch.cuda.reset_peak_memory_stats()
        if phase:
            rows = full_index(model, grid, psi, args.hessian_tol, args.hessian_iter, mu, args.index_modes,
                              x0=None if warm is None else torch.cat(list(warm.values())),
                              save=run.data("hessian_modes.pt"))
        else:
            rows = parity_index(model, grid, psi, args.hessian_tol, args.hessian_iter, x0=warm,
                                save=run.data("hessian_modes.pt"), mu=mu)
        out.update(hessian=rows, hessian_peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
        run.save_json("tube_saddle_refine.json", out)
        if phase:
            run.result(index=rows["index"], complete=rows["complete"], certified=rows["certified"],
                       eigenvalues=rows["eigenvalues"])
        else:
            run.result(index=rows["index"], certified=rows["certified"], even=rows["even"]["eigenvalues"],
                       odd=rows["odd"]["eigenvalues"])


def _uniform(model, grid, psi_U, cells, N, ensemble):
    """Energy of the tiled uniform state with N atoms and, for the grand ensemble, its mu."""
    psi = dipgpe.normalize(grid, psi_U.repeat(1, 1, cells).to(grid.dtype), N)
    mu = float(dipgpe.chemical_potential(model, psi)) if ensemble == "grand" else None
    return float(dipgpe.energy(model, psi)), mu


def _dz(grid, f):
    return grid.ifft(1j * grid.k[2] * grid.fft(f))


def asym(args, run):
    L, Np, psi_U, _, cells = endpoints(args)
    grid, model = tube(args, L, Np, DTYPES[args.dtypes])
    folder = next(dipgpe.runs.runs_dir().glob(args.from_run + "*"))
    psi_s = torch.load(folder / "data" / "saddle.pt").to("cuda", grid.dtype)
    warm = stored_modes(folder, grid)
    if warm is None:
        raise SystemExit(f"{folder.name} has no Hessian modes")
    N = float(args.density * PER_L * args.cells * L)
    E_U, mu = _uniform(model, grid, psi_U, args.cells, N, args.ensemble)
    psi_u = dipgpe.normalize(grid, psi_U.repeat(1, 1, args.cells).to(grid.dtype), N)
    scale = float(dipgpe.norm(grid, psi_s - psi_u)) ** 0.5
    amplitudes = [float(a) * scale for a in args.amplitudes.split(",")]
    t0 = time.time()

    def log(row):
        print(f"  a = {row['a'] / scale:.4f} (x norm), beta = {row['beta']:+.4e}, residual {row['residual']:.1e}, "
              f"converged {row['converged']}  ({time.time() - t0:.0f} s)", flush=True)
    torch.cuda.reset_peak_memory_stats()
    psi, info = break_symmetry(model, psi_s, warm["odd"][0].real.to(grid.dtype), amplitudes, mu=mu,
                               deflate=[_dz(grid, psi_s).real.to(grid.dtype)], max_newton=args.max_newton,
                               callback=log)
    pol = info["polish"] or {}
    dN = pol.get("N", float("nan")) - N
    barrier = pol.get("energy", float("nan")) - E_U - (0.0 if mu is None else mu * dN)
    odd_part = float((psi - mirror(psi)).abs().norm() / psi.abs().norm())
    print(f"converged {info['converged']}, a = {info['a'] and info['a'] / scale}, barrier {barrier:.6f} hbar omega, "
          f"delta_N = {dN:.2f}, residual {pol.get('residual')}, full residual {pol.get('full_residual')}, "
          f"multiplier {pol.get('multipliers')}, odd part {odd_part:.3e}, "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    line = ((psi.abs() ** 2).sum((0, 1)) * grid.dx[0] * grid.dx[1]).double().cpu()
    torch.save(psi.cpu(), run.data("saddle.pt"))
    out = {"cells": cells, "N": N, "ensemble": args.ensemble, "mu": mu, "from_run": folder.name, "E_uniform": E_U,
           "amplitude_unit": scale, "scan": info["scan"], "bracket": info["bracket"], "a": info["a"],
           "break_converged": info["converged"], "barrier": barrier, "delta_N": dN, "odd_part": odd_part,
           "residuals": pol.get("residuals", [float("nan")]), "full_residual": pol.get("full_residual"),
           "multipliers": pol.get("multipliers"), "seconds": time.time() - t0,
           "line_density_per_um": (line / PER_L).tolist(), "dz_um": UNITS.microns(grid.dx[2])}
    run.save_json("tube_saddle_asym.json", out)
    run.result(barrier=barrier, delta_N=dN, converged=info["converged"], a=info["a"], odd_part=odd_part,
               residual=pol.get("residual"), seconds=time.time() - t0)
    if not info["converged"]:
        return
    torch.cuda.empty_cache()
    rows = full_index(model, grid, psi, args.hessian_tol, args.hessian_iter, mu, args.index_modes,
                      x0=torch.cat([warm["even"], warm["odd"]]), save=run.data("hessian_modes.pt"))
    out.update(hessian=rows)
    run.save_json("tube_saddle_asym.json", out)
    run.result(index=rows["index"], complete=rows["complete"], certified=rows["certified"],
               eigenvalues=rows["eigenvalues"])


def connect(args, run):
    if args.ensemble != "grand":
        raise SystemExit("connect needs --ensemble grand")
    L, Np, psi_U, _, cells = endpoints(args)
    grid, model = tube(args, L, Np, DTYPES[args.dtypes])
    folder = next(dipgpe.runs.runs_dir().glob(args.from_run + "*"))
    psi_s = torch.load(folder / "data" / "saddle.pt").to("cuda", grid.dtype)
    modes = stored_modes(folder, grid)
    if modes is None:
        raise SystemExit(f"{folder.name} has no Hessian modes")
    symmetric = "full" not in modes
    v = (modes["even"] if symmetric else modes["full"])[0].real.to(grid.dtype)
    N = float(args.density * PER_L * args.cells * L)
    E_U, mu = _uniform(model, grid, psi_U, args.cells, N, args.ensemble)
    omega_U = E_U - mu * N
    psi_u = dipgpe.normalize(grid, psi_U.repeat(1, 1, args.cells).to(grid.dtype), N)
    dev = psi_s - psi_u
    scale = float(dipgpe.norm(grid, dev)) ** 0.5
    v = v / float(dipgpe.norm(grid, v)) ** 0.5
    if float((v.conj() * dev).sum().real) < 0:       # +v grows the nucleus
        v = -v
    E_s = float(dipgpe.energy(model, psi_s))
    omega_s = E_s - mu * float(dipgpe.norm(grid, psi_s))
    barrier = omega_s - omega_U
    dz = UNITS.microns(grid.dx[2])

    def geometry(psi):
        line = ((psi.abs() ** 2).sum((0, 1)) * grid.dx[0] * grid.dx[1]).double().cpu() / PER_L
        return nucleus(line.tolist(), dz, args.nz_cell, args.density)

    geo_s = geometry(psi_s)
    print(f"saddle {folder.name}: barrier {barrier:.4f} hbar omega, FWHM {geo_s['fwhm_um']:.1f} um, "
          f"contrast {geo_s['contrast']:.3f}, {'even sector' if symmetric else 'phase condition'}", flush=True)
    sides = {}
    for side, sign in (("shrink", -1.0), ("grow", 1.0)):
        if side not in args.sides.split(","):
            continue
        t0 = time.time()
        trace = []

        def watch(it, psi, info, side=side, trace=trace, t0=t0):
            geo = geometry(psi)
            d = info["omega"] - omega_U
            trace.append({"iteration": it, "omega_minus_uniform": d, "N_minus_uniform": info["N"] - N,
                          "residual": info["residual"], "fwhm_um": geo["fwhm_um"], "contrast": geo["contrast"],
                          "droplets": geo["droplets"], "seconds": time.time() - t0})
            if it % (10 * args.every) == 0:
                print(f"  {side} {it}: Omega - Omega_U = {d:.4f}, FWHM {geo['fwhm_um']:.1f} um, contrast "
                      f"{geo['contrast']:.3f}, residual {info['residual']:.1e}  ({time.time() - t0:.0f} s)",
                      flush=True)
            if side == "shrink":
                return geo["contrast"] < 0.02 and abs(d) < 1e-3 * barrier
            return geo["fwhm_um"] >= geo_s["fwhm_um"] + args.grow_cells * args.cell_um and d < barrier
        psi, info = descend(model, psi_s + sign * args.eps * scale * v, mu=mu, max_iter=args.max_descent,
                            tau=args.tau, sector=(lambda f: 0.5 * (f + mirror(f))) if symmetric else None,
                            deflate=None if symmetric else [_dz(grid, psi_s).real.to(grid.dtype)],
                            callback=watch, callback_every=args.every)
        torch.save(psi.cpu(), run.data(f"{side}.pt"))
        geo = geometry(psi)
        final = {"omega_minus_uniform": info["omega"] - omega_U, "N_minus_uniform": info["N"] - N,
                 "residual": info["residual"], **geo}
        if side == "shrink":
            reached = geo["contrast"] < 0.02 and abs(final["omega_minus_uniform"]) < 1e-3 * barrier
        else:
            reached = info["stopped"]
        sides[side] = {"reached": bool(reached), "iterations": info["iterations"], "converged": info["converged"],
                       "stopped": info["stopped"], "final": final, "trace": trace}
        print(f"  {side}: {'reached' if reached else 'NOT reached'} after {info['iterations']} iterations, "
              f"Omega - Omega_U = {info['omega'] - omega_U:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    run.save_json("tube_saddle_connect.json", {"from_run": folder.name, "symmetric": symmetric, "mu": mu,
                                                "barrier": barrier, "saddle": geo_s, "eps": args.eps,
                                                "tau": args.tau, "sides": sides})
    run.result(barrier=barrier, **{f"{k}_reached": v["reached"] for k, v in sides.items()},
               connected=all(v["reached"] for v in sides.values()) and len(sides) == 2)


def staircase(args, run):
    if args.ensemble != "grand":
        raise SystemExit("staircase needs --ensemble grand")
    L, Np, psi_U, _, cells = endpoints(args)
    grid, model = tube(args, L, Np, DTYPES[args.dtypes])

    def stored(rid, name):
        return torch.load(next(dipgpe.runs.runs_dir().glob(rid + "*")) / "data" / f"{name}.pt").to("cuda", grid.dtype)

    cur, prev = stored(args.from_run, args.state), stored(args.prev_run, args.prev_state)
    N = float(args.density * PER_L * args.cells * L)
    E_U, mu = _uniform(model, grid, psi_U, args.cells, N, args.ensemble)
    omega_U = E_U - mu * N
    mask = (0.5 * (1 + torch.tanh(grid.x64[2] / (0.5 * L)))).to(grid.real_dtype)   # z > 0: the right front
    fracs = [float(f) for f in args.amplitudes.split(",")]

    def omega(psi):
        return float(dipgpe.energy(model, psi)) - mu * float(dipgpe.norm(grid, psi)) - omega_U

    t0 = time.time()
    print(f"start {args.from_run}/{args.state}: Omega - Omega_U = {omega(cur):.4f}; previous "
          f"{args.prev_run}/{args.prev_state}: {omega(prev):.4f}", flush=True)
    rows = []
    out = {"from_run": args.from_run, "state": args.state, "prev_run": args.prev_run, "prev_state": args.prev_state,
           "mu": mu, "start_omega_minus_uniform": omega(cur), "prev_omega_minus_uniform": omega(prev), "rungs": rows}
    for k in range(1, args.rungs + 1):
        step = cur - prev
        unit = float(dipgpe.norm(grid, step)) ** 0.5

        def log(r, k=k, unit=unit):
            print(f"  rung {k}: a = {r['a'] / unit:.3f} (x last step), beta = {r['beta']:+.3e}, "
                  f"Omega - Omega_U = {r['energy'] - mu * r['N'] - omega_U:.4f}, residual {r['residual']:.1e}, "
                  f"converged {r['converged']}  ({time.time() - t0:.0f} s)", flush=True)
        nxt, info = break_symmetry(model, cur, (mask * step).to(grid.dtype), [f * unit for f in fracs], mu=mu,
                                   deflate=[_dz(grid, cur).real.to(grid.dtype)], max_newton=args.max_newton,
                                   callback=log)
        row = {"rung": k, "converged": info["converged"], "a_over_step": info["a"] and info["a"] / unit,
               "step_norm": unit, "bracket": info["bracket"],
               "scan": [{**r, "omega_minus_uniform": r["energy"] - mu * r["N"] - omega_U} for r in info["scan"]]}
        rows.append(row)
        if not info["converged"]:
            print(f"rung {k}: no next stationary state found", flush=True)
            break
        pol = info["polish"]
        row.update(omega_minus_uniform=pol["energy"] - mu * pol["N"] - omega_U, delta_N=pol["N"] - N,
                   residual=pol["residual"], full_residual=pol["full_residual"], multipliers=pol["multipliers"])
        torch.save(nxt.cpu(), run.data(f"rung{k}.pt"))
        torch.cuda.empty_cache()
        row["hessian"] = full_index(model, grid, nxt, args.hessian_tol, args.hessian_iter, mu, args.index_modes)
        print(f"rung {k}: Omega - Omega_U = {row['omega_minus_uniform']:.4f}, delta_N = {row['delta_N']:.1f}, "
              f"index {row['hessian']['index']}  ({time.time() - t0:.0f} s)", flush=True)
        run.save_json("tube_saddle_staircase.json", out)
        prev, cur = cur, nxt
    run.save_json("tube_saddle_staircase.json", out)
    run.result(states=[{k: r.get(k) for k in ("rung", "converged", "omega_minus_uniform", "delta_N")}
                       | {"index": r.get("hessian", {}).get("index")} for r in rows], seconds=time.time() - t0)


def full_index(model, grid, psi, tol, iters, mu, n, x0=None, save=None):
    """Lowest ``n`` Hessian eigenvalues in the full space (no parity sectors), the
    translation d psi/dz deflated; for states without a mirror symmetry (asym) or
    refined with a phase condition (refine --sector none)."""
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    lam, modes, hinfo = constrained_hessian(model, psi, n_modes=n, deflate=[_dz(grid, psi)], tol=tol,
                                            max_iter=iters, mu=mu, x0=x0)
    lam = lam.tolist()
    parity = [float((m - mirror(m)).abs().norm() / (m + mirror(m)).abs().norm().clamp_min(1e-300)) for m in modes]
    rows = {"eigenvalues": lam, "residuals": hinfo["residual"], "iterations": hinfo["iterations"],
            "odd_over_even_norm": parity, "seconds": time.time() - t1,
            "index": sum(v < 0 for v in lam), "complete": lam[-1] > 0,
            "certified": all(r < 1 for r in hinfo["residual"]),
            "peak_GiB": torch.cuda.max_memory_allocated() / 2 ** 30}
    print(f"  Hessian (full space): {lam} residuals {hinfo['residual']} ({hinfo['iterations']} it, "
          f"{time.time() - t1:.0f} s, peak {rows['peak_GiB']:.1f} GiB): index {rows['index']}"
          f"{'' if rows['complete'] else ' (incomplete: all computed eigenvalues negative)'}", flush=True)
    if save is not None:
        torch.save({"full": modes.to(torch.complex64).cpu()}, save)
    return rows


def hessian(args, run):
    L, Np, psi_U, _, _ = endpoints(args)
    grid, model = tube(args, L, Np, DTYPES[args.dtypes])
    folder = next(dipgpe.runs.runs_dir().glob(args.from_run + "*"))
    psi = torch.load(folder / "data" / "saddle.pt").to("cuda", grid.dtype)
    _, mu = _uniform(model, grid, psi_U, args.cells, float(args.density * PER_L * args.cells * L), args.ensemble)
    torch.cuda.reset_peak_memory_stats()
    rows = parity_index(model, grid, psi, args.hessian_tol, args.hessian_iter, x0=stored_modes(folder, grid),
                        save=run.data("hessian_modes.pt"), mu=mu)
    rows["peak_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"  index {rows['index']}, peak {rows['peak_GiB']:.1f} GiB", flush=True)
    run.save_json("tube_saddle_hessian.json", {"from_run": folder.name, **rows})
    run.result(index=rows["index"], certified=rows["certified"], even=rows["even"]["eigenvalues"],
               odd=rows["odd"]["eigenvalues"])


def _record(folder):
    import json
    f = folder / "tube_saddle_refine.json"
    return json.loads((f if f.exists() else folder / "tube_saddle_asym.json").read_text(encoding="utf-8"))


def _index_of(folder):
    """Morse index of a refine or asym run: its own Hessian, or a hessian run made from it."""
    import json
    out = _record(folder)
    h = out.get("hessian")
    if isinstance(h, dict):
        return h["index"], h
    for hf in sorted(dipgpe.runs.runs_dir().glob("*_tube_saddle_hessian/tube_saddle_hessian.json")):
        hj = json.loads(hf.read_text(encoding="utf-8"))
        if hj.get("from_run") == folder.name and "even" in hj:
            even, odd = hj["even"]["eigenvalues"], hj["odd"]["eigenvalues"]
            return sum(v < 0 for v in even) + sum(v < 0 for v in odd), hj
    return None, None


def nucleus(line, dz, nz_cell, density):
    """Geometry of a nucleus centred at z = 0 from the line density (atoms per micron)."""
    import numpy as np
    n = np.asarray(line)
    h = nz_cell // 2
    ext = np.concatenate([n[-h:], n, n[:h]])
    win = np.lib.stride_tricks.sliding_window_view(ext, 2 * h + 1)
    C = (win.max(1) - win.min(1)) / (win.max(1) + win.min(1))
    q = n.size // 4
    background = float(np.concatenate([n[:q // 2], n[-q // 2:]]).mean())
    half = C >= 0.5 * C.max()
    top = background + 0.5 * (n.max() - background)
    peaks = int(((n > np.roll(n, 1)) & (n >= np.roll(n, -1)) & (n > top)).sum())
    c, d = n.size // 2, max(nz_cell // 4, 1)
    centre = "droplet" if n[c] > n[c - d] and n[c] > n[c + d] else "gap"
    return {"contrast": float(C.max()), "fwhm_um": float(half.sum() * dz), "droplets": peaks,
            "background_minus_mean": background - density, "centre": centre}


def summary(args, run):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for rid in args.runs.split(","):
        folder = next(dipgpe.runs.runs_dir().glob(rid + "*"))
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        prm = meta.get("params") or meta.get("parameters")
        out = _record(folder)
        index, _ = _index_of(folder)
        geo = nucleus(out["line_density_per_um"], out["dz_um"], prm["nz_cell"], prm["density"])
        rows.append({"run": folder.name, "a_s": prm["a"], "cells": prm["cells"], "dx": prm["dx"],
                     "ensemble": prm.get("ensemble", "canonical"), "delta_N": out.get("delta_N"),
                     "L_perp": prm["L_perp"], "barrier": out["barrier"], "residual": out["residuals"][-1],
                     "index": index, **geo, "line": out["line_density_per_um"], "dz_um": out["dz_um"]})
    rows.sort(key=lambda r: (r["ensemble"], r["cells"], r["a_s"], r["centre"]))
    print(f"{'a_s':>7} {'cells':>5} {'ensemble':>9} {'centre':>8} {'barrier':>10} {'index':>5} {'contrast':>8} "
          f"{'FWHM um':>8} {'drops':>5} {'bg-n /um':>8} {'residual':>9}")
    for r in rows:
        print(f"{r['a_s']:7.3f} {r['cells']:5d} {r['ensemble']:>9} {r['centre']:>8} {r['barrier']:10.4f} "
              f"{str(r['index']):>5} "
              f"{r['contrast']:8.3f} {r['fwhm_um']:8.1f} {r['droplets']:5d} {r['background_minus_mean']:8.2f} "
              f"{r['residual']:9.1e}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for cells in sorted({r["cells"] for r in rows}):
        for centre, mark in (("gap", "s"), ("droplet", "o")):
            sel = [r for r in rows if r["cells"] == cells and r["centre"] == centre]
            if sel:
                ax1.plot([r["a_s"] for r in sel], [r["barrier"] for r in sel], mark + "-",
                         label=f"{cells} cells, {centre} at z = 0")
    ax1.axvline(89.83, ls=":", c="gray")
    ax1.set_xlabel("a_s (a0)")
    ax1.set_ylabel("E_saddle - E_uniform (hbar omega)")
    ax1.legend()
    for k, r in enumerate([r for r in rows if r["cells"] == min(x["cells"] for x in rows)]):
        import numpy as np
        z = (np.arange(len(r["line"])) - len(r["line"]) // 2) * r["dz_um"]
        ax2.plot(z, np.asarray(r["line"]) + 4000 * k, lw=0.8, label=f"{r['a_s']:.2f}")
    ax2.set_xlabel("z (um)")
    ax2.set_ylabel("line density (1/um), offset")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run.file("tube_saddle_summary.png"), dpi=150)
    run.save_json("tube_saddle_summary.json", [{k: v for k, v in r.items() if k != "line"} for r in rows])
    run.result(rows=[{k: r[k] for k in ("a_s", "cells", "ensemble", "barrier", "delta_N", "index", "fwhm_um",
                                         "contrast")} for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cost", "pilot", "refine", "asym", "connect", "staircase", "hessian",
                                     "summary"])
    ap.add_argument("--runs", type=str, default="", help="summary: comma-separated refine run ids")
    ap.add_argument("--a", type=float, default=None, help="scattering length (a0); required except for summary")
    ap.add_argument("--density", type=float, default=6250.0, help="mean density (atoms per micron)")
    ap.add_argument("--cells", type=int, default=32, help="tube length in crystal cells")
    ap.add_argument("--cell-um", type=float, default=2.76)
    ap.add_argument("--nz-cell", type=int, default=17, help="grid points per cell along z")
    ap.add_argument("--patch", type=int, default=8, help="crystal cells of the seed (even)")
    ap.add_argument("--dx", type=float, default=0.25)
    ap.add_argument("--L-perp", type=float, default=24.0)
    ap.add_argument("--images", type=str, default="9", help="number of images (cost: comma-separated list)")
    ap.add_argument("--dtypes", type=str, default="c64", help="c64 or c128 (cost: comma-separated list)")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=12, help="cost: iterations per measurement")
    ap.add_argument("--minutes", type=float, default=10.0, help="pilot: time budget")
    ap.add_argument("--from-run", type=str, default=None, help="pilot: start from the string of this run ...")
    ap.add_argument("--end-image", type=str, default=None, help="... between its image 0 and this image "
                    "(auto: the first image past the highest one)")
    ap.add_argument("--phase", choices=["gap", "droplet"], default="gap",
                    help="nucleus seed: a gap or a droplet at z = 0")
    ap.add_argument("--image", type=int, default=None, help="refine: image of a stored string (default: its "
                    "saddle); a refine run as --from-run gives its saddle (continuation in a_s, embedding in L)")
    ap.add_argument("--max-newton", type=int, default=8, help="refine: Newton steps")
    ap.add_argument("--grow", type=int, default=0, help="refine: lengthen the start nucleus by this many cells "
                    "on each side (2 x this many more droplets)")
    ap.add_argument("--modes-from", type=str, default=None, help="refine: run whose Hessian modes warm-start the "
                    "index (default: --from-run)")
    ap.add_argument("--hessian", action="store_true", help="refine: lowest two Hessian eigenvalues (Morse index)")
    ap.add_argument("--hessian-tol", type=float, default=1e-3)
    ap.add_argument("--hessian-iter", type=int, default=300)
    ap.add_argument("--climb-tol", type=float, default=1e-2, help="pilot: start climbing below this normal "
                    "gradient (negative: never)")
    ap.add_argument("--every", type=int, default=50, help="pilot: log every this many iterations")
    ap.add_argument("--gpu-fraction", type=float, default=0.85, help="cap on the GPU memory this process may use")
    ap.add_argument("--amplitudes", type=str, default="0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5",
                    help="asym: odd amplitudes in units of the nucleus norm; staircase: in units of the last step")
    ap.add_argument("--index-modes", type=int, default=3,
                    help="asym, refine --sector none: Hessian eigenvalues in the full space")
    ap.add_argument("--eps", type=float, default=0.05, help="connect: start offset, in units of the nucleus norm")
    ap.add_argument("--tau", type=float, default=0.5, help="connect: descent step")
    ap.add_argument("--max-descent", type=int, default=8000, help="connect: descent iterations per side")
    ap.add_argument("--sides", type=str, default="shrink,grow", help="connect: which sides to descend")
    ap.add_argument("--prev-run", type=str, default=None, help="staircase: run with the state before --from-run")
    ap.add_argument("--prev-state", type=str, default="saddle", help="staircase: stored state name in --prev-run")
    ap.add_argument("--rungs", type=int, default=2, help="staircase: stationary states to find")
    ap.add_argument("--state", type=str, default="saddle",
                    help="refine: the stored state of --from-run to start from (saddle, or shrink/grow of a connect run)")
    ap.add_argument("--grow-cells", type=float, default=3.0,
                    help="connect: the growing side counts as reached once the nucleus is this many cells longer")
    ap.add_argument("--sector", choices=["even", "none"], default="even",
                    help="refine: impose z -> -z symmetry, or none (translation fixed by a phase condition)")
    ap.add_argument("--ensemble", choices=["canonical", "grand"], default="canonical",
                    help="refine, hessian: fixed N, or fixed mu of the uniform state (grand canonical)")
    args = ap.parse_args()
    if args.patch % 2:
        raise SystemExit("--patch must be even (patch centred at z = 0)")
    if args.a is None and args.mode != "summary":
        raise SystemExit("--a is required")
    if torch.cuda.is_available():
        # beyond this PyTorch raises OutOfMemoryError instead of letting the Windows driver
        # spill GPU allocations into system memory (a 64-cell Hessian filled 30 GB of RAM, ISSUES C3-11)
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    with dipgpe.runs.Run(f"tube_saddle_{args.mode}", vars(args)) as run:
        {"cost": cost, "pilot": pilot, "refine": refine, "asym": asym, "connect": connect, "staircase": staircase,
         "hessian": hessian, "summary": summary}[args.mode](args, run)


if __name__ == "__main__":
    main()
