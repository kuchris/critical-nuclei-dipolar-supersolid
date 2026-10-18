"""Figures of paper C3a (docs/paper_C3a_outline.md) from stored runs; no simulation.

Fig. 1  concept: the critical nucleus at 90.0 a0 (density slice, line density),
        the a_s window, the energy along the string path at 90.0 a0.
Fig. 2  shape of the nucleus: line densities across the window; peak contrast and
        length (FWHM of the one-cell contrast, tube_saddle.nucleus) against a_s.
Fig. 3  barrier E_b(a_s) at fixed mu (= the L -> infinity barrier) from the roton
        spinodal to the coexistence edge, with 2 sigma(a_s); inset E_b^(2/3) near a_c.
Fig. 4  roton end: E_b / eps^(3/2), contrast / eps^(1/2) and FWHM eps^(1/2) against
        eps = a_s - a_c (a_c from the fit in the inset of Fig. 3).
Fig. 5  finite tube: schematic of the nucleus in tubes of length L and 2L; fixed-N
        barrier minus fixed-mu barrier against L, with Delta N^2 / (2 kappa L) (no
        fit); the difference between two tube lengths against its prediction (no
        fixed-mu reference); Delta N(a_s).
Fig. 6  schematic of the droplets sliding inside the nucleus and the energy along
        that sliding; lowest Hessian eigenvalues of the gap- and droplet-centred
        nuclei: sliding (odd sector) and breathing (even sector); for states refined
        without parity sectors the second and first eigenvalues in the full space.
Fig. 7  coexistence edge: stationary states along the growth path at 90.272 and
        90.273 (Omega against atoms expelled) with the descents from the saddles;
        the deficit 2 sigma - E_b against the drive; inset sigma(a_s).
Fig. 8  connectivity: schematic of the saddle nudged both ways; energy released along
        the descents (shrinking and growing side), nucleus size (growing) and
        contrast (shrinking).

Journal style as C1: drawn at the printed size (two-column width of REVTeX, 510 pt),
8 pt STIX text, no lettering below 6 pt.

Every number is read from the run records named below (docs/RESULTS.md, C3-2);
runs marked invalid (dipgpe.runs note --invalid) are refused. The script runs as a
run record itself (figures as PDF and PNG, and the numbers they show, result.json);
the PDFs and SOURCES.txt (the runs each figure reads) are also written into
paper/C3a/figures for compiling the manuscript. Two fields are drawn (Figs. 1a and
2); where the run records come without their data folders, as in the public release,
they are read from the reduced data set data_C3a (tools/export_c3a_data.py).

Usage:
  uv run python benchmarks/c3a_figures.py [--figs 1,2,3,4,5,6,7,8] [--out paper/C3a/figures]
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import dipgpe  # noqa: E402
from tube_coexistence import PER_L  # noqa: E402
from tube_saddle import nucleus  # noqa: E402
from tube_supersolid import UNITS  # noqa: E402

A_ROT = 89.845                     # BdG roton spinodal on this grid (run 20261005-143507)
A_ROT_RUN = "20261005-143507"
WINDOW = (90.237, 90.274)          # coexistence at fixed n = 6250/um (run 20261008-123945)
COEX = "20261008-123945"           # window, coexisting densities and uniform branch mu(n) at 90.256 a0
CELL_UM, NZ_CELL, L_PERP, DX, DENSITY = 2.76, 17, 24.0, 0.25, 6250.0
DATA = ROOT / "data_C3a" / "states"  # reduced data set; read only where a run's data folder is absent

# fixed-mu saddles: a_s, run (or (staircase run, rung)), kind; 89.86-89.88 at 64 cells
BARRIERS = [
    (89.86, "20261018-040436", "gap"), (89.87, "20261018-045759", "gap"), (89.88, "20261018-045934", "gap"),
    (89.90, "20261017-235057", "gap"), (89.95, "20261018-035921", "gap"), (90.00, "20261018-035941", "gap"),
    (90.05, "20261017-235035", "droplet"), (90.10, "20261018-040000", "droplet"),
    (90.15, "20261018-040049", "droplet"), (90.20, "20261017-234923", "droplet"),
    (90.21, "20261018-040105", "droplet"), (90.22, "20261018-040120", "droplet"),
    (90.23, "20261018-040140", "droplet"), (90.24, "20261018-040235", "gap"),
    (90.245, "20261018-040255", "gap"), (90.25, "20261017-235127", "gap"), (90.255, "20261018-040316", "gap"),
    (90.26, "20261018-040807", "gap"), (90.265, "20261018-023307", "gap"),
    (90.27, "20261018-024346", "asymmetric"), (90.272, ("20261018-103344", 1), "second rung"),
    (90.273, "20261018-110707", "second rung"),
]
SIGMA = [(90.25, "20261008-130517"), (90.256, "20261008-132115"), (90.265, "20261017-231702"),
         (90.274, "20261017-231854")]
STRING_90 = "20261016-173115"      # pilot string at 90.0 a0 (complex64, 9 images)
NEWTON_90 = 14.4926                # Newton-refined saddle of that string (fixed N, 32 cells; RESULTS)
NUCLEUS = ("20261018-035941", "saddle")  # critical nucleus at 90.00 a0 (fixed mu, gap-centred)

# Fig. 2: fixed-mu saddles whose line densities are drawn (89.88 at 64 cells)
SHAPES = [(89.88, "20261018-045934"), (90.05, "20261017-235035"), (90.20, "20261017-234923"),
          (90.25, "20261017-235127"), (90.27, "20261018-024346"), (90.273, "20261018-110707")]
# Fig. 5: a_s -> (fixed-mu run, {cells: fixed-N run of the same nucleus})
FINITE_L = {
    89.90: ("20261017-235057", {32: "20261017-202122", 64: "20261017-215133", 128: "20261017-221037"}),
    90.05: ("20261017-235035", {32: "20261017-205917", 64: "20261017-220616", 128: "20261017-221204"}),
    90.20: ("20261017-234923", {32: "20261017-210726", 64: "20261017-220704", 128: "20261017-221259"}),
    90.25: ("20261017-235127", {32: "20261017-232532", 64: "20261017-235519", 128: "20261017-235618"}),
    90.26: ("20261018-040807", {32: "20261017-232856", 64: "20261017-235743", 128: "20261018-000133"}),
}
# Fig. 6: saddles with parity-sector Hessians, 32 cells (fixed N: continuation in a_s; the fixed-N
# gap-centred state at 90.265 has moved to a longer state and is left out), and fixed-mu states
# indexed in the full space (no parity sectors)
EIG_N = {"gap": ["20261017-201116", "20261017-201441", "20261017-201801", "20261017-202122", "20261017-202634",
                 "20261017-203201", "20261017-203720", "20261017-232211", "20261017-232415", "20261017-232532",
                 "20261017-232653", "20261017-232856"],
         "droplet": ["20261017-205917", "20261017-210420", "20261017-210601", "20261017-210726",
                     "20261017-223225", "20261017-223326", "20261017-223431", "20261017-223553",
                     "20261017-233407", "20261017-233605", "20261017-233723", "20261017-233848",
                     "20261017-234030"]}
EIG_MU = {"gap": ["20261018-035537", "20261017-235127", "20261018-040316"],
          "droplet": ["20261018-040000", "20261017-234923", "20261018-040140", "20261018-002400",
                      "20261018-004021", "20261018-052437"]}
EIG_FULL = [("20261018-040807", "gap"), ("20261018-023307", "gap"), ("20261018-024346", "asymmetric"),
            ("20261018-110707", "second rung")]
# Fig. 7: stationary states along the growth path (saddle, minimum) and states of other branches
# reached by staircase steps that jumped (ISSUES C3-16); descents from the saddles
STAIRS = {90.272: [("20261018-052437", "saddle"), ("20261018-054330", "minimum"),
                   (("20261018-103344", 1), "saddle"), (("20261018-103344", 2), "minimum"),
                   (("20261018-105238", 1), "other"), (("20261018-105238", 2), "other")],
          90.273: [("20261018-110707", "saddle"), ("20261018-110931", "minimum"),
                   (("20261018-111328", 1), "other")]}
TRACES = {90.272: ("20261018-053925", ("shrink", "grow")), 90.273: ("20261018-111156", ("shrink",))}
EDGE = (90.25, 90.255, 90.26, 90.265, 90.27, 90.272, 90.273)
# Fig. 8: a_s, connect run of the shrinking side, connect run of the growing side
CONNECT = [(89.88, "20261018-044820", "20261018-044820"), (90.20, "20261018-042650", "20261018-042650"),
           (90.27, "20261018-045542", "20261018-050130"), (90.272, "20261018-053925", "20261018-053925")]

STYLE = {"gap": ("s", "#1f77b4"), "droplet": ("o", "#2ca02c"), "asymmetric": ("D", "#9467bd"),
         "second rung": ("^", "#d62728")}
W2 = 510 / 72.27                   # two-column width of REVTeX (\textwidth, in): figures drawn at printed size


def style():
    """Journal style as C1: figures drawn at their printed size, 8 pt STIX (Times-like) text."""
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0, "lines.markersize": 3.2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False, "legend.handlelength": 1.6,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.formatter.use_mathtext": True, "pdf.fonttype": 42})


def title(ax, label, text=""):
    """Bold panel label and a short description, left-aligned above the panel."""
    ax.set_title(rf"$\mathbf{{{label}}}$ {text}".rstrip(), loc="left", pad=3)


def folder(rid):
    f = next(dipgpe.runs.runs_dir().glob(rid + "*"))
    meta = json.loads((f / "meta.json").read_text(encoding="utf-8"))
    if any(n.get("invalid") for n in meta.get("notes", [])):
        raise SystemExit(f"{f.name} is marked invalid")
    return f


def load(rid, name):
    return json.loads((folder(rid) / name).read_text(encoding="utf-8"))


def a_of(rid):
    meta = load(rid, "meta.json")
    return (meta.get("params") or meta.get("parameters"))["a"]


def record(rid):
    """Refine or asym record of a stationary state."""
    f = folder(rid)
    name = "tube_saddle_refine.json" if (f / "tube_saddle_refine.json").exists() else "tube_saddle_asym.json"
    return json.loads((f / name).read_text(encoding="utf-8"))


def field(rid, name):
    """A stored state: from the run's data folder or, where that is absent (the public release), from the
    reduced data set (tools/export_c3a_data.py), which keeps the real field in double precision."""
    f = folder(rid) / "data" / f"{name}.pt"
    if f.exists():
        return torch.load(f)
    g = next(DATA.glob(f"*_{rid}_{name}"), None)
    if g is None:
        raise SystemExit(f"{rid} {name}: no field; unpack the data archive data_C3a.zip into {DATA.parent.parent}")
    return torch.from_numpy(np.load(g / "psi.npz")["psi"])


def rung(ref):
    return load(ref[0], "tube_saddle_staircase.json")["rungs"][ref[1] - 1]


def barrier(ref):
    """Omega - Omega_U (fixed mu) or E - E_U (fixed N) of a run, or of (staircase run, rung)."""
    if isinstance(ref, tuple):
        return rung(ref)["omega_minus_uniform"]
    return load(ref, "result.json")["barrier"]


def delta_n(ref):
    return rung(ref)["delta_N"] if isinstance(ref, tuple) else record(ref)["delta_N"]


def line_density(ref):
    """Line density (atoms per um) and its grid step (um)."""
    if isinstance(ref, tuple):                         # staircase states keep only the field
        rid, name = ref[0], f"rung{ref[1]}"
        saved = DATA.parent / "line_densities.json"    # the reduced data keep the line densities of all of them
        if not (folder(rid) / "data" / f"{name}.pt").exists() and saved.exists():
            rec = json.loads(saved.read_text(encoding="utf-8"))[f"{rid}_{name}"]
            return np.asarray(rec["line_density_per_um"]), rec["dz_um"]
        psi = field(rid, name)
        return (psi.abs() ** 2).sum((0, 1)).double().numpy() * DX * DX / PER_L, CELL_UM / NZ_CELL
    rec = record(ref)
    return np.asarray(rec["line_density_per_um"]), rec["dz_um"]


def geometry(ref):
    line, dz = line_density(ref)
    return nucleus(line, dz, NZ_CELL, DENSITY)


def hessian(rid):
    """(even, odd) eigenvalues by parity sectors, or (None, eigenvalues) in the full space."""
    h = record(rid)["hessian"]
    return (h["even"]["eigenvalues"], h["odd"]["eigenvalues"]) if "even" in h else (None, h["eigenvalues"])


def sigma(rid):
    return load(rid, "result.json")["sigma_hbar_omega"]


def two_sigma(a):
    """2 sigma(a_s), linear between the measured points (32 cells)."""
    xs, ys = zip(*[(a_, 2 * sigma(r)) for a_, r in SIGMA])
    return np.interp(a, xs, ys)


def roton_fit():
    """a_c and the line E_b^(2/3)(a_s) through the three 64-cell saddles next to the spinodal."""
    a = np.array([r[0] for r in BARRIERS[:3]])
    E = np.array([barrier(r[1]) for r in BARRIERS[:3]])
    p = np.polyfit(a, E ** (2 / 3), 1)
    return float(-p[1] / p[0]), p


def coexistence():
    """kappa = dn/dmu of the uniform tube at n (per um per hbar omega, from its mu(n) at 90.256 a0) and the
    drive Delta omega(a_s) = dn_coex |dn_coex/da_s| (a_edge - a_s) / kappa (hbar omega per um; ISSUES C3-13)."""
    d = load(COEX, "tube_coexistence.json")
    w = d["window"]
    row = next(r for r in d["per_a"] if abs(r["a_s"] - 90.256) < 1e-9)
    n = np.array([u["n"] for u in row["uniform"]])
    mu = np.array([u["mu"] for u in row["uniform"]])
    dmu_dn = float(np.interp(w["n_ref"], 0.5 * (n[1:] + n[:-1]), np.diff(mu) / np.diff(n)))

    def drive(a):
        return w["dn_coex_mean"] * abs(w["slope_n_per_a0"]) * (w["a_from_uniform_side"] - np.asarray(a)) * dmu_dn
    return 1.0 / dmu_dn, drive


def by_kind(ax, x, y, kinds, ms=3.5, **kw):
    for kind, (m, col) in STYLE.items():
        sel = [i for i, k in enumerate(kinds) if k == kind]
        if sel:
            ax.plot(np.asarray(x)[sel], np.asarray(y)[sel], m, color=col, ms=ms,
                    label=kind + " centred" if kind in ("gap", "droplet") else kind, **kw)


def window(ax):
    ax.axvspan(*WINDOW, color="#9fd3c7", alpha=0.4, lw=0)
    ax.axvline(A_ROT, color="0.5", ls=":", lw=0.8)


def save(fig, out, name):
    """PDF into every directory of ``out`` (the run record first, then paper/C3a/figures); PNG for viewing
    into the run record only."""
    fig.tight_layout(pad=0.3)
    for d in out:
        fig.savefig(Path(d) / f"{name}.pdf")
    fig.savefig(Path(out[0]) / f"{name}.png", dpi=200)
    plt.close(fig)


def sources():
    """The runs each figure reads, for figures/SOURCES.txt."""
    def ids(refs):
        return sorted({r if isinstance(r, str) else f"{r[0]} (rung {r[1]})" for r in refs})
    barriers = [r[1] for r in BARRIERS]
    finite = [m for m, _ in FINITE_L.values()] + [r for _, runs in FINITE_L.values() for r in runs.values()]
    return {
        "fig1_concept": ids([NUCLEUS[0], STRING_90, COEX]),
        "fig2_shape": ids([r for _, r in SHAPES] + barriers),
        "fig3_barrier": ids(barriers + [r for _, r in SIGMA]),
        "fig4_roton_end": ids([r[1] for r in BARRIERS if r[0] <= 90.20]),
        "fig5_finite_tube": ids(finite + barriers + [COEX]),
        "fig6_eigenvalues": ids(EIG_N["gap"] + EIG_N["droplet"] + EIG_MU["gap"] + EIG_MU["droplet"]
                                + [r for r, _ in EIG_FULL]),
        "fig7_edge": ids([ref for states in STAIRS.values() for ref, _ in states] + [r for r, _ in TRACES.values()]
                         + [r[1] for r in BARRIERS if r[0] in EDGE] + [r for _, r in SIGMA] + [COEX]),
        "fig8_connectivity": ids([rid for row in CONNECT for rid in row[1:]]),
    }


def write_sources(out, run_dir):
    head = (f"Figures of paper C3a, written by benchmarks/c3a_figures.py into this folder and into its run record "
            f"runs/{run_dir.name} (same PDFs, plus PNGs; the numbers the figures show are in its result.json, the "
            f"git commit in its meta.json). Journal style: drawn at the printed two-column width (510 pt), 8 pt "
            f"STIX text. Every figure reads only the stored runs listed below; runs marked invalid are refused. "
            f"Constants in the script: roton spinodal {A_ROT} a0 (run {A_ROT_RUN}), metastable window "
            f"{WINDOW[0]}-{WINDOW[1]} a0 (run {COEX}), E_b = {NEWTON_90} at 90.0 a0 (Newton, fixed N; RESULTS).")
    lines = textwrap.wrap(head, 100) + [""]
    for name, refs in sources().items():
        lines += textwrap.wrap(f"{name}.pdf ({len(refs)} runs): " + ", ".join(refs), 100,
                               subsequent_indent="    ") + [""]
    for d in out:
        (Path(d) / "SOURCES.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ---- schematic panels (drawn shapes, no data) ----

def schematic_axes(fig, spec, xmax, ymax):
    ax = fig.add_subplot(spec)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.set_anchor("W")                                 # left-aligned, like the panels below
    ax.axis("off")
    return ax


def arrow(ax, p, q, rad=0.0, ls="-", color="0.25"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, ls=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def catmull(pts, n=12):
    """Points on a Catmull-Rom spline through ``pts`` (smooth schematic curves)."""
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = (np.asarray(p, float) for p in P[i - 1:i + 3])
        for t in np.arange(n) / n:
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out.append(np.asarray(pts[-1], float))
    return np.array(out)


def tubes_schematic(ax):
    """Fig. 5(a): the same nucleus in tubes of length L and 2L; the atoms it expels stay in the tube."""
    for x0, x1, y, label in ((1.55, 4.05, 1.85, "$L$"), (0.3, 5.3, 0.75, "$2L$")):
        ax.add_patch(FancyBboxPatch((x0, y - 0.25), x1 - x0, 0.5, boxstyle="round,pad=0,rounding_size=0.25",
                                    fc="0.93", ec="0.55", lw=0.6))
        for dx in (-0.17, 0.0, 0.17):
            ax.add_patch(Ellipse((2.8 + dx, y), 0.1, 0.36, fc="#1D9E75", ec="none"))
        for s in (-1, 1):
            arrow(ax, (2.8 + 0.35 * s, y), (2.8 + 0.85 * s, y), color="0.45")
        ax.text(x1 + 0.12, y, label, va="center")
    lines = (r"fixed $N$: the $|\Delta N|$ atoms the nucleus expels stay in the tube;",
             r"the background gains $|\Delta N|/L$, which costs $\Delta N^2/(2\kappa L)$",
             r"fixed $\mu$: they leave to a reservoir; $\Delta\Omega$ is the $L\to\infty$ barrier",
             r"(c): $E_b(L) - E_b(2L) = \Delta N^2/(4\kappa L)$, no reference needed")
    for k, line in enumerate(lines):
        ax.text(6.0, 2.2 - 0.5 * k, line, va="center", fontsize=7)


def sliding_schematic(ax):
    """Fig. 6(a): the droplets slide inside the nucleus envelope; energy along that sliding."""
    gap_col, drop_col, asym_col = STYLE["gap"][1], STYLE["droplet"][1], STYLE["asymmetric"][1]
    bags = ((1.5, gap_col, "gap-centred", ((-0.63, 0.5), (-0.21, 0.5), (0.21, 0.5), (0.63, 0.5))),
            (5.1, drop_col, "droplet-centred", ((-0.84, 0.28), (-0.42, 0.5), (0, 0.5), (0.42, 0.5), (0.84, 0.28))))
    for cx, col, label, drops in bags:
        ax.add_patch(FancyBboxPatch((cx - 1.2, 1.65), 2.4, 0.85, boxstyle="round,pad=0,rounding_size=0.35",
                                    fill=False, ec=col, ls="--", lw=0.8))
        for dx, h in drops:
            ax.add_patch(Ellipse((cx + dx, 2.075), 0.16, h, fc="0.55", ec="none"))
        ax.plot([cx, cx], [1.5, 1.6], color=col, lw=1.2)
        ax.text(cx, 1.25, label, ha="center", va="center", fontsize=7, color=col)
    arrow(ax, (2.85, 2.075), (3.75, 2.075))
    ax.text(3.3, 2.8, "slide droplets\nby half a period", ha="center", va="center", fontsize=6.5)
    cases = (("90.05–90.23", lambda u: np.cos(2 * np.pi * u), "droplet-centred\nin the valley"),
             ("90.24–90.26", lambda u: -np.cos(2 * np.pi * u), "gap-centred\nin the valley"),
             ("90.27", lambda u: np.cos(4 * np.pi * u), "valley in between:\nasymmetric"))
    for k, (label, E, caption) in enumerate(cases):
        x0 = 7.0 + 2.4 * k

        def X(v, x0=x0):
            return x0 + 1.7 * np.asarray(v)

        def Y(v, E=E):
            return 2.0 + 0.32 * E(np.asarray(v))
        u = np.linspace(0, 1, 81)
        ax.plot(X(u), Y(u), color="0.4", lw=0.9)
        ax.plot(X([0, 1]), Y([0, 1]), "s", color=gap_col, ms=3.5)
        ax.plot(X(0.5), Y(0.5), "o", color=drop_col, ms=4)
        if k == 2:
            ax.plot(X([0.25, 0.75]), Y([0.25, 0.75]), "D", color=asym_col, ms=3.5)
        ax.text(X(0.5), 2.75, f"$a_s$ = {label}", ha="center", va="center", fontsize=6.5)
        for v, name in ((0, "gap"), (0.5, "droplet"), (1, "gap")):
            ax.text(X(v), 1.45, name, ha="center", va="center", fontsize=6, color="0.35")
        ax.text(X(0.5), 0.95, caption, ha="center", va="center", fontsize=6.5)
    ax.text(10.3, 0.3, "valley: sliding eigenvalue > 0;  hump: < 0", ha="center", va="center", fontsize=6.5,
            color="0.3")


def rolling_schematic(ax):
    """Fig. 8(a): the saddle nudged both ways (near the edge the growing side stops in a pinned state: Fig. 7(a))."""
    pts = ((0.0, 0.0), (0.12, 0.02), (0.30, 0.55), (0.45, 1.0), (0.62, 0.55), (0.80, -0.55), (0.95, -1.6))
    ball = 0.45

    def X(u):
        return 0.6 + 9.0 * np.asarray(u)

    def Y(e):
        return 1.45 + 0.85 * np.asarray(e)
    c = catmull(pts)
    E = dict(pts)
    ax.plot(X(c[:, 0]), Y(c[:, 1]), color="0.4", lw=0.9)
    bx, by = float(X(ball)), float(Y(E[ball])) + 0.15
    ax.add_patch(Circle((bx, by), 0.13, fc="#D85A30", ec="none", zorder=3))
    arrow(ax, (bx - 0.2, by + 0.02), (float(X(0.18)), float(Y(0.22)) + 0.2), rad=0.35, ls="--")
    arrow(ax, (bx + 0.2, by + 0.02), (float(X(0.7)), float(Y(0.1)) + 0.25), rad=-0.35)
    ax.text(float(X(0.0)), float(Y(0.0)) - 0.32, "uniform", fontsize=6.5)
    ax.text(float(X(0.95)) + 0.1, float(Y(pts[-1][1])) + 0.15, "crystal\ngrows", fontsize=6.5)
    ax.text(float(X(0.5)), 2.85, "saddle", ha="center", va="center", fontsize=7)
    ax.text(10.4, 1.55, "near the edge the growing side\nstops in a pinned state: Fig. 7(a)", va="center", fontsize=6.5,
            color="0.3")


def stationary_path(points, n=40):
    """Schematic curve through stationary points: a smoothstep between neighbours, so that each point is an
    extremum with zero slope and no other extremum lies between them."""
    t = np.arange(n) / n
    s = 3 * t ** 2 - 2 * t ** 3
    xs = [x0 + (x1 - x0) * t for (x0, _), (x1, _) in zip(points[:-1], points[1:])] + [[points[-1][0]]]
    ys = [y0 + (y1 - y0) * s for (_, y0), (_, y1) in zip(points[:-1], points[1:])] + [[points[-1][1]]]
    return np.concatenate(xs), np.concatenate(ys)


def staircase_schematic(ax):
    """Fig. 7(a): growth beyond the first saddle at 90.272 a0 drawn as an energy landscape. Positions and heights
    are schematic (the order of the stationary energies is kept, not their spacing); the values are printed with
    the computed line densities below. Only the computed stationary states are extrema of the solid curve; what
    follows M2 was not computed (dashed)."""
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.7, 3.15)
    U, S1, M1, S2, M2 = (0.7, 0.0), (3.3, 2.2), (4.3, 2.0), (5.5, 2.85), (7.6, 0.95)
    ax.plot(*stationary_path([(0.0, 0.0), U, S1, M1, S2, M2]), color="0.25", lw=1.2)
    ax.plot(*stationary_path([M2, (8.7, 1.3)]), color="0.25", lw=1.2, ls=(0, (3, 2)))
    ax.text(8.8, 1.3, "further growth\n(not followed)", fontsize=6.5, color="0.35", va="center")
    ax.plot(*stationary_path([U, (3.6, 2.3), (8.4, 0.5)]), color="0.6", lw=0.8, ls=(0, (2, 2)))
    ax.text(6.4, 1.05, "without pinning\n(conceptual, not calculated)", fontsize=6.5, color="0.5", ha="right",
            va="top")
    red = STYLE["second rung"][1]
    ax.plot(*U, "s", ms=4.5, color="0.35", zorder=4)
    for (x, y), m in ((S1, "^"), (M1, "o"), (S2, "^"), (M2, "o")):
        ax.plot(x, y, m, ms=5, mfc=red if m == "^" else "white", mec=red, mew=1.0, zorder=4)
    ax.text(U[0], U[1] - 0.17, "uniform", fontsize=6.5, ha="center", va="top")
    ax.text(S1[0] - 0.13, S1[1] + 0.02, "S$_1$", fontsize=7, ha="right", va="bottom")
    ax.text(M1[0], M1[1] - 0.16, "M$_1$", fontsize=7, ha="center", va="top")
    ax.text(S2[0] + 0.13, S2[1] + 0.02, "S$_2$", fontsize=7, ha="left", va="bottom")
    ax.text(M2[0], M2[1] - 0.16, "M$_2$", fontsize=7, ha="center", va="top")
    brown = "#9a3412"                                              # barriers between successive pinned states
    ax.plot([S1[0] + 0.1, M1[0] - 0.08], [S1[1]] * 2, color=brown, lw=0.5, ls=":")
    ax.plot([M1[0] + 0.05, S2[0] - 0.1], [S2[1]] * 2, color=brown, lw=0.5, ls=":")
    arrow(ax, (M1[0] - 0.1, M1[1] + 0.06), (M1[0] - 0.1, S1[1]), color=brown)
    arrow(ax, (M1[0] + 0.07, M1[1] + 0.06), (M1[0] + 0.07, S2[1]), color=brown)
    ax.text(M1[0] - 0.18, S1[1] + 0.13, "0.022", fontsize=6.5, color=brown, ha="right", va="bottom")
    ax.text(M1[0] + 0.15, 2.55, "0.144", fontsize=6.5, color=brown, ha="left")
    blue = "#1f4e79"                                               # the barrier from the uniform state
    ax.plot([0.3, S2[0] - 0.12], [S2[1]] * 2, color=blue, lw=0.5, ls=":")
    ax.add_patch(FancyArrowPatch((0.3, 0.0), (0.3, S2[1]), arrowstyle="<|-|>", mutation_scale=7, lw=0.8,
                                 color=blue, shrinkA=0, shrinkB=0))
    for y in (1.35, 1.47):                                         # not to scale
        ax.plot([0.21, 0.39], [y - 0.06, y + 0.06], color=blue, lw=0.8)
    ax.text(0.42, 2.72, "$E_b = 91.416$: highest saddle\non this path, from the uniform state", fontsize=6.5,
            color=blue, ha="left", va="top")
    ax.text(5.0, -0.62, r"energies schematic, not to scale; $\Omega - \Omega_U$ of the computed states "
                        r"($\hbar\omega_\perp$) with their line densities below", fontsize=6.5, color="0.3",
            ha="center")


def fig1(out):
    um = UNITS.microns(1.0)
    psi = field(*NUCLEUS).numpy()
    nx, ny, nz = psi.shape
    rho = np.abs(psi) ** 2
    slice_yz = rho[nx // 2, :, :]                      # dipoles along y: in the plane
    line = rho.sum((0, 1)) * DX * DX                   # atoms per l
    line_um = line / um                                # atoms per um
    z = (np.arange(nz) - nz // 2) * CELL_UM / NZ_CELL
    y = (np.arange(ny) - ny // 2) * DX * um

    fig = plt.figure(figsize=(W2, 4.1))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(slice_yz, origin="lower", aspect="auto", cmap="magma",
              extent=[z[0], z[-1], y[0], y[-1]])
    ax.set_xlim(-30, 30)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$z$ ($\mu$m)")
    ax.set_ylabel(r"$y$ ($\mu$m)")
    title(ax, "(a)", "critical nucleus at 90.0 $a_0$ (density, $x=0$)")
    ax.annotate("", xy=(-25, 3.4), xytext=(-25, 1.4), arrowprops=dict(arrowstyle="->", color="w", lw=0.8))
    ax.text(-23.5, 2.4, r"$\mathbf{d}\parallel y$", color="w", fontsize=7, va="center")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(z, line_um / 1e3)
    ax.axhline(6.25, color="0.5", ls="--", lw=0.8)
    ax.set_xlim(-30, 30)
    ax.set_xlabel(r"$z$ ($\mu$m)")
    ax.set_ylabel(r"$n(z)$ ($10^3\,\mu$m$^{-1}$)")
    title(ax, "(b)", "line density; dashed: uniform superfluid")

    ax = fig.add_subplot(2, 2, 3)
    ax.axvspan(89.68, A_ROT, color="0.85")
    ax.axvspan(A_ROT, WINDOW[1], color="#cfe8e2")
    ax.axvspan(WINDOW[0], WINDOW[1], color="#9fd3c7")
    ax.axvspan(WINDOW[1], 90.42, color="0.95")
    ax.text(89.762, 0.55, "superfluid\nunstable", ha="center", va="center", fontsize=7)
    ax.text(90.04, 0.55, "superfluid metastable:\ncrystal by nucleation", ha="center", va="center", fontsize=7)
    ax.text(90.347, 0.55, "superfluid\nstable", ha="center", va="center", fontsize=7)
    ax.text(90.2555, 0.62, "coexistence", ha="center", va="center", fontsize=6, rotation=90)
    ax.annotate("", xy=(89.72, 0.15), xytext=(90.40, 0.15), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(90.06, 0.2, r"ramp $a_s$ down", ha="center", fontsize=7)
    ax.set_xlim(89.68, 90.42)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(top=False)
    ax.set_xticks([A_ROT, WINDOW[1]])
    ax.set_xticklabels([r"$a_{\rm rot}^*$" + "\n89.845", "edge\n90.274"])
    ax.set_xlabel(r"$a_s$ ($a_0$)")
    title(ax, "(c)", r"the metastable window at $6250\,\mu$m$^{-1}$")

    ax = fig.add_subplot(2, 2, 4)
    pj = json.loads((folder(STRING_90) / "tube_saddle_pilot.json").read_text(encoding="utf-8"))
    E = np.array(pj["energies_minus_E0"])[:-1]         # last image: the fixed end of this segment
    s = np.linspace(0, 1, len(E))
    ax.plot(s, E, "o-", color="0.3")
    c = pj["saddle"]
    ax.plot(s[c], E[c], "o", color="#d62728", ms=5)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.annotate(r"critical nucleus ($E_b$)", xy=(s[c], E[c]), xytext=(0.05, -140), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(0.0, -45, "uniform", fontsize=7)
    ax.text(0.35, -330, "nucleus grows:\ncrystal", fontsize=7)
    ax.set_xlabel("path coordinate (string images)")
    ax.set_ylabel(r"$E - E_U$ ($\hbar\omega_\perp$)")
    title(ax, "(d)", f"string path at 90.0 $a_0$ ($E_b$ = {NEWTON_90:.1f})")
    save(fig, out, "fig1_concept")


def fig2(out):
    fig = plt.figure(figsize=(W2, 3.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[:, 0])
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.85, len(SHAPES)))
    for k, ((a_s, ref), col) in enumerate(zip(SHAPES, colors)):
        line, dz = line_density(ref)
        z = (np.arange(line.size) - line.size // 2) * dz
        ax.plot(z, (line - DENSITY) / 1e3 + 10 * k, lw=0.8, color=col)
        ax.text(-43, 10 * k + 1.5, f"{a_s:g} $a_0$", color=col, fontsize=7)
    ax.plot([41, 41], [2.5, 7.5], color="k", lw=1.2)
    ax.text(40, 5, r"$5\times10^3\,\mu$m$^{-1}$", ha="right", va="center", fontsize=6.5)
    ax.set_xlim(-44, 44)
    ax.set_yticks([])
    ax.tick_params(right=False)
    ax.set_xlabel(r"$z$ ($\mu$m)")
    ax.set_ylabel(r"$n(z) - n$ (curves offset)")
    title(ax, "(a)", r"line density of the critical nucleus (fixed $\mu$)")
    rows = [(a, kind, geometry(ref)) for a, ref, kind in BARRIERS]
    a = [r[0] for r in rows]
    kinds = [r[1] for r in rows]
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, 1], sharex=bx)
    by_kind(bx, a, [r[2]["contrast"] for r in rows], kinds)
    by_kind(cx, a, [r[2]["fwhm_um"] for r in rows], kinds)
    for axis in (bx, cx):
        window(axis)
    bx.set_ylabel("peak contrast")
    title(bx, "(b)", "depth")
    bx.legend(fontsize=6, loc="lower right")
    plt.setp(bx.get_xticklabels(), visible=False)
    cx.set_ylabel(r"FWHM ($\mu$m)")
    cx.set_xlabel(r"$a_s$ ($a_0$)")
    title(cx, "(c)", "length")
    save(fig, out, "fig2_shape")
    return {a_: (round(g["contrast"], 3), round(g["fwhm_um"], 1), g["droplets"]) for a_, _, g in rows}


def fig3(out):
    a = np.array([r[0] for r in BARRIERS])
    E = np.array([barrier(r[1]) for r in BARRIERS])
    kinds = [r[2] for r in BARRIERS]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    sa = np.array([s_ for s_, _ in SIGMA])
    for axis in (ax, bx):
        by_kind(axis, a, E, kinds)
        axis.plot(sa, two_sigma(sa), "-", color="#ff7f0e", lw=1.2, label=r"$2\sigma(a_s)$")
    window(ax)
    bx.axvline(WINDOW[1], color="0.5", ls="--", lw=0.8)
    bx.text(WINDOW[1] - 0.0007, 73.5, "edge", ha="right", fontsize=7, color="0.4")
    ax.set_xlabel(r"$a_s$ ($a_0$)")
    ax.set_ylabel(r"$E_b$ ($\hbar\omega_\perp$)")
    ax.set_xlim(89.83, 90.285)
    title(ax, "(a)", r"barrier at fixed $\mu$ ($L\to\infty$)")
    ax.legend(loc="lower right")
    ins = ax.inset_axes([0.1, 0.5, 0.34, 0.36])
    ac, p = roton_fit()
    near = a <= 90.0
    ins.plot(a[near], E[near] ** (2 / 3), "s", ms=3, color="#1f77b4")
    xs = np.linspace(ac, 90.0, 50)
    ins.plot(xs, np.polyval(p, xs), "-", lw=0.8, color="0.3")
    ins.axvline(A_ROT, color="0.5", ls=":", lw=0.8)
    ins.set_xlabel(r"$a_s$ ($a_0$)", fontsize=7, labelpad=1)
    ins.set_ylabel(r"$E_b^{2/3}$", fontsize=7, labelpad=1)
    ins.tick_params(labelsize=6)
    ins.text(ac + 0.005, 3.0, f"$a_c$ = {ac:.3f}", fontsize=6.5)
    bx.set_xlim(90.235, 90.276)
    bx.set_ylim(72, 97)
    bx.set_xlabel(r"$a_s$ ($a_0$)")
    title(bx, "(b)", "near the coexistence edge")
    save(fig, out, "fig3_barrier")
    return dict(zip(a.tolist(), E.tolist())), ac


def fig4(out):
    ac, _ = roton_fit()
    rows = [r for r in BARRIERS if r[0] <= 90.20]
    eps = np.array([r[0] for r in rows]) - ac
    E = np.array([barrier(r[1]) for r in rows])
    geo = [geometry(r[1]) for r in rows]
    C = np.array([g["contrast"] for g in geo])
    W = np.array([g["fwhm_um"] for g in geo])
    kinds = [r[2] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    panels = ((E / eps ** 1.5, r"$E_b\,/\,\varepsilon^{3/2}$ ($\hbar\omega_\perp\,a_0^{-3/2}$)", "(a)", "barrier"),
              (C / eps ** 0.5, r"contrast$\,/\,\varepsilon^{1/2}$ ($a_0^{-1/2}$)", "(b)", "depth"),
              (W * eps ** 0.5, r"FWHM$\,\varepsilon^{1/2}$ ($\mu$m$\,a_0^{1/2}$)", "(c)", "length"))
    for ax, (y, label, letter, text) in zip(axs, panels):
        by_kind(ax, eps, y, kinds)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.15 * y.max())
        ax.set_xlabel(r"$\varepsilon = a_s - a_c$ ($a_0$)")
        ax.set_ylabel(label)
        title(ax, letter, text)
    axs[0].legend(fontsize=6, loc="lower left")
    save(fig, out, "fig4_roton_end")
    # last entry: E_b w / A^2 with A the contrast, independent of a_c (4c/3 in the amplitude equation)
    return {round(float(e), 4): (round(float(x), 1), round(float(c), 3), round(float(w), 2), round(float(v), 0))
            for e, x, c, w, v in zip(eps, E / eps ** 1.5, C / eps ** 0.5, W * eps ** 0.5, E * W / C ** 2)}


def fig5(out):
    kappa, _ = coexistence()
    fig = plt.figure(figsize=(W2, 3.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 2.0], width_ratios=[1.2, 1, 1])
    sx = schematic_axes(fig, gs[0, :], 14, 2.6)
    tubes_schematic(sx)
    title(sx, "(a)", r"a nucleus in a tube of fixed $N$ (schematic)")
    ax, bx, cx = (fig.add_subplot(gs[1, k]) for k in range(3))
    colors = plt.get_cmap("plasma")(np.linspace(0.0, 0.8, len(FINITE_L)))
    ratios = {}
    for col, (a_s, (mu_run, runs)) in zip(colors, FINITE_L.items()):
        omega, dN = barrier(mu_run), delta_n(mu_run)
        cells = sorted(runs)                           # 32, 64, 128: each tube twice the last
        L = np.array(cells) * CELL_UM
        E = np.array([barrier(runs[c]) for c in cells])
        Ls = np.geomspace(70, 450, 20)
        ax.loglog(L, E - omega, "o", color=col, ms=3.5)
        ax.loglog(Ls, dN ** 2 / (2 * kappa * Ls), "-", color=col, lw=0.8)
        # E(L) - E(2L) against Delta N^2/(4 kappa L): no fixed-mu reference (its own finite-size error,
        # ~3e-4 at 89.90 a0, would dominate the long-tube excess of 0.01)
        diff = (E[:-1] - E[1:]) / (dN ** 2 / (4 * kappa * L[:-1]))
        bx.plot(1 / L[:-1], diff, "o-", color=col, ms=3.5, lw=0.8, label=f"{a_s:.2f}")
        ratios[a_s] = [round(float(r), 4) for r in diff]
    ax.set_xticks(np.array(sorted(runs)) * CELL_UM)
    ax.set_xticklabels([f"{c * CELL_UM:.0f}" for c in sorted(runs)])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"tube length $L$ ($\mu$m)")
    ax.set_ylabel(r"$E_b(N, L) - \Delta\Omega$ ($\hbar\omega_\perp$)")
    title(ax, "(b)", r"fixed $N$ minus fixed $\mu$")
    bx.legend(title=r"$a_s$ ($a_0$), also in (b)", fontsize=6, title_fontsize=6, loc="upper left")
    bx.axhline(1, color="0.5", lw=0.6)
    bx.set_xlim(0, 0.012)
    bx.set_xlabel(r"$1/L$ ($\mu$m$^{-1}$), $L$ the shorter tube")
    bx.set_ylabel(r"$[E_b(L) - E_b(2L)]\,/\,[\Delta N^2/(4\kappa L)]$", fontsize=7)
    title(bx, "(c)", "two lengths compared")
    dns = {r[0]: delta_n(r[1]) for r in BARRIERS}
    by_kind(cx, list(dns), [-v for v in dns.values()], [r[2] for r in BARRIERS])
    window(cx)
    cx.set_xlabel(r"$a_s$ ($a_0$)")
    cx.set_ylabel(r"$-\Delta N$ (atoms expelled)")
    title(cx, "(d)", r"$\Delta N$ at fixed $\mu$")
    save(fig, out, "fig5_finite_tube")
    return kappa, ratios, {k: round(v, 1) for k, v in dns.items()}


def broken(x, *ys, gap=0.06):
    """Insert NaN where consecutive x are farther apart than ``gap``, so lines join only neighbours."""
    x = np.asarray(x, float)
    cut = np.where(np.diff(x) > gap)[0] + 1
    return [np.insert(np.asarray(v, float), cut, np.nan) for v in (x, *ys)]


def fig6(out):
    fig = plt.figure(figsize=(W2, 4.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 2.0])
    sx = schematic_axes(fig, gs[0, :], 14, 3.1)
    sliding_schematic(sx)
    title(sx, "(a)", "sliding the droplets inside the nucleus (schematic)")
    ax, bx = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for kind in ("gap", "droplet"):
        m, col = STYLE[kind]
        for runs, fixed_n in ((EIG_N[kind], True), (EIG_MU[kind], False)):
            pts = sorted((a_of(r), *hessian(r)) for r in runs)
            a, odd, even = broken([p[0] for p in pts], [1e3 * p[2][0] for p in pts], [1e2 * p[1][0] for p in pts])
            style_ = dict(color=col, mfc=col if fixed_n else "white", ms=3.5, lw=0.7, ls="-" if fixed_n else "none")
            ax.plot(a, odd, m, label=f"{kind} centred, fixed " + ("$N$" if fixed_n else r"$\mu$"), **style_)
            bx.plot(a, even, m, **style_)
    labelled = set()
    for rid, kind in EIG_FULL:
        m, col = STYLE[kind]
        lam = hessian(rid)[1]
        label = None if kind in labelled or kind == "gap" else f"{kind} (full space)"
        labelled.add(kind)
        ax.plot(a_of(rid), 1e3 * lam[1], m, color=col, mfc="white", ms=4.5, mew=1.0, label=label)
        bx.plot(a_of(rid), 1e2 * lam[0], m, color=col, mfc="white", ms=4.5, mew=1.0)
    for axis in (ax, bx):
        axis.axhline(0, color="0.5", lw=0.6)
        window(axis)
        axis.set_xlabel(r"$a_s$ ($a_0$)")
    ax.set_ylabel(r"sliding eigenvalue ($10^{-3}\,\hbar\omega_\perp$)")
    title(ax, "(b)", "sliding mode (lowest odd)")
    ax.legend(fontsize=6, loc="lower left")
    bx.set_ylabel(r"breathing eigenvalue ($10^{-2}\,\hbar\omega_\perp$)")
    title(bx, "(c)", "growth mode (lowest even)")
    save(fig, out, "fig6_eigenvalues")


def fig7(out):
    _, drive = coexistence()
    fig = plt.figure(figsize=(W2, 5.0))
    gs = fig.add_gridspec(3, 10, height_ratios=[1.3, 0.62, 2.5])
    sx = fig.add_subplot(gs[0, :])
    staircase_schematic(sx)
    title(sx, "(a)", r"growth beyond the first saddle at $90.272\,a_0$")
    path = [ref for ref, kind in STAIRS[90.272] if kind != "other"]          # S1, M1, S2, M2 along the path
    names = ("S$_1$ first saddle", "M$_1$ pinned minimum", "S$_2$ second saddle", "M$_2$ next minimum")
    thumbs = [("uniform superfluid", None, 0.0)] + [(n, r, barrier(r)) for n, r in zip(names, path)]
    for k, (name, ref, omega) in enumerate(thumbs):
        th = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        if ref is None:
            z, dn = np.array([-30.0, 30.0]), np.zeros(2)
        else:
            line, dz = line_density(ref)
            z = (np.arange(line.size) - line.size // 2) * dz
            dn = (line - DENSITY) / 1e3
        th.plot(z, dn, lw=0.6, color="0.4" if ref is None else STYLE["droplet"][1])
        th.set_xlim(-30, 30)
        th.set_ylim(-5, 5)
        th.set_xticks([])
        th.set_yticks([])
        th.tick_params(top=False, right=False)
        for s in th.spines.values():
            s.set_linewidth(0.4)
            s.set_color("0.6")
        th.set_title(f"{name}\n{omega:.3f}" if ref is not None else f"{name}\n0", fontsize=6.5, pad=2)
    ax, bx = fig.add_subplot(gs[2, :6]), fig.add_subplot(gs[2, 6:])
    colors = {90.272: "#d62728", 90.273: "#1f77b4"}
    for a_s, states in STAIRS.items():
        col = colors[a_s]
        rid, sides = TRACES[a_s]
        c = load(rid, "tube_saddle_connect.json")
        start = (-delta_n(c["from_run"][:15]), c["barrier"])          # the saddle the descents leave
        for side in sides:
            tr = c["sides"][side]["trace"]
            x = [start[0]] + [-t["N_minus_uniform"] for t in tr]
            y = [start[1]] + [t["omega_minus_uniform"] for t in tr]
            # the nudge along the unstable mode and the refilling of the background at fixed mu (first 50
            # iterations) move Delta N quickly before the nucleus itself moves: dashed
            k = 1 + next(i for i, t in enumerate(tr) if t["iteration"] >= 50)
            ax.plot(x[:k + 1], y[:k + 1], "--", color=col, lw=0.8, alpha=0.6)
            ax.plot(x[k:], y[k:], "-", color=col, lw=0.8, alpha=0.6)
        on_path = [ref for ref, kind in states if kind != "other"]
        ax.plot([-delta_n(r) for r in on_path], [barrier(r) for r in on_path], ":", color=col, lw=0.8)
        for ref, kind in states:
            m = {"saddle": "^", "minimum": "o", "other": "x"}[kind]
            ax.plot(-delta_n(ref), barrier(ref), m, color=col, ms=4, mfc=col if kind == "saddle" else "white")
    for ref, label, (dx, dy, ha) in zip(path, ("S$_1$", "M$_1$", "S$_2$", "M$_2$"),
                                        ((-5, 4, "right"), (0, -11, "center"), (0, 5, "center"), (5, -9, "left"))):
        ax.annotate(label, (-delta_n(ref), barrier(ref)), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=7, color=colors[90.272])
    ax.set_xlim(450, 1300)
    ax.set_ylim(89.4, 92.7)
    ax.set_xlabel(r"$-\Delta N$ (atoms expelled by the nucleus)")
    ax.set_ylabel(r"$\Omega - \Omega_U$ ($\hbar\omega_\perp$)")
    title(ax, "(b)", "computed states near the edge")
    handles = [Line2D([], [], color=colors[a], lw=2, label=f"{a:g} $a_0$") for a in colors] + [
        Line2D([], [], marker="^", color="0.3", ls="none", ms=4, label="saddle (index 1)"),
        Line2D([], [], marker="o", color="0.3", mfc="white", ls="none", ms=4, label="minimum (index 0)"),
        Line2D([], [], marker="x", color="0.3", ls="none", ms=4, label="other branch (index 1)"),
        Line2D([], [], color="0.3", alpha=0.6, lw=0.8, label="descent from a saddle"),
        Line2D([], [], color="0.3", alpha=0.6, lw=0.8, ls="--", label="its first 50 steps"),
        Line2D([], [], color="0.3", lw=0.8, ls=":", label="order along the path")]
    ax.legend(handles=handles, fontsize=6, loc="lower left")
    Eb = {r[0]: barrier(r[1]) for r in BARRIERS}
    kinds = {r[0]: r[2] for r in BARRIERS}
    a = np.array(EDGE)
    deficit = two_sigma(a) - np.array([Eb[x] for x in a])
    w = drive(a)
    by_kind(bx, w, deficit, [kinds[x] for x in a])
    # 2 sigma is uncertain by about +-0.3: the 32-cell, 64-cell and reference-free sigma differ by up to 0.13
    # (paper Appendix D); shown at the last point, where it matters
    bx.errorbar(w[-1], deficit[-1], yerr=0.3, color=STYLE["second rung"][1], capsize=2, lw=0.8)
    for x, y, a_s in zip(w, deficit, a):
        bx.annotate(f"{a_s:g}", (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6)
    bx.axhline(0, color="0.5", lw=0.6)
    bx.set_xlim(0, 1.15 * w.max())
    bx.set_ylim(-1.0, 1.15 * deficit.max())               # with pinning the limit is 2 sigma + Delta P
    bx.set_xlabel(r"drive $\Delta\omega$ ($\hbar\omega_\perp\,\mu$m$^{-1}$)")
    bx.set_ylabel(r"$2\sigma - E_b$ ($\hbar\omega_\perp$)")
    title(bx, "(c)", "deficit to the thin-wall limit")
    ins = bx.inset_axes([0.22, 0.62, 0.25, 0.33])         # clear of the main axis and of the 90.26 point
    ins.plot([s_ for s_, _ in SIGMA], [sigma(r) for _, r in SIGMA], "o-", ms=3, lw=0.8, color="#ff7f0e")
    ins.set_xlim(90.246, 90.278)                           # the end points inside the frame
    ins.set_ylim(46.0, 48.0)
    ins.set_xticks([90.25, 90.27])
    ins.set_yticks([46.5, 47.0, 47.5])
    ins.tick_params(labelsize=6)
    ins.set_xlabel(r"$a_s$ ($a_0$)", fontsize=6.5, labelpad=1)
    ins.set_ylabel(r"$\sigma$ ($\hbar\omega_\perp$)", fontsize=6.5)
    ins.yaxis.set_label_coords(-0.36, 0.5)                 # left of the tick labels
    save(fig, out, "fig7_edge")
    return {float(x): (round(float(v), 3), round(float(d), 3)) for x, v, d in zip(a, w, deficit)}


def fig8(out):
    fig = plt.figure(figsize=(W2, 4.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 2.0], width_ratios=[1.4, 1, 1])
    sx = schematic_axes(fig, gs[0, :], 14, 3.1)
    rolling_schematic(sx)
    title(sx, "(a)", "the saddle nudged both ways (schematic)")
    axs = [fig.add_subplot(gs[1, k]) for k in range(3)]
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.85, len(CONNECT)))
    final = {}
    for (a_s, shrink, grow), col in zip(CONNECT, colors):
        for rid, side, ls in ((shrink, "shrink", "--"), (grow, "grow", "-")):
            c = load(rid, "tube_saddle_connect.json")
            tr = [t for t in c["sides"][side]["trace"] if t["iteration"] > 0]
            it = np.array([t["iteration"] for t in tr])
            drop = c["barrier"] - np.array([t["omega_minus_uniform"] for t in tr])
            axs[0].loglog(it, np.clip(drop, 1e-5, None), ls, color=col,
                          label=f"{a_s:g} $a_0$" if side == "grow" else None)
            if side == "grow":
                axs[1].loglog(it, [-t["N_minus_uniform"] for t in tr], ls, color=col)
            else:
                axs[2].semilogx(it, [t["contrast"] for t in tr], ls, color=col)
            final[f"{a_s:g} {side}"] = (int(it[-1]), round(float(drop[-1]), 4))
        axs[0].axhline(c["barrier"], color=col, ls=":", lw=0.6)
    handles, _ = axs[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", ls="--", label="shrinking side"),
                Line2D([], [], color="0.3", label="growing side"),
                Line2D([], [], color="0.3", ls=":", lw=0.6, label=r"$\Omega_s - \Omega_U$")]
    axs[0].set_xlabel("descent iteration")
    axs[0].set_ylabel(r"$\Omega_s - \Omega$ ($\hbar\omega_\perp$)")
    title(axs[0], "(b)", "energy released")
    axs[0].set_ylim(1e-2, 3e5)                         # room for the legend above the curves
    axs[0].legend(handles=handles, fontsize=6, loc="upper left", ncol=2, columnspacing=1.0)
    axs[1].set_xlabel("descent iteration")
    axs[1].set_ylabel(r"$-\Delta N$ (atoms expelled)")
    title(axs[1], "(c)", "growing side: nucleus size")
    axs[2].set_xlabel("descent iteration")
    axs[2].set_ylabel("peak contrast")
    title(axs[2], "(d)", "shrinking side")
    save(fig, out, "fig8_connectivity")
    return final


FIGS = "1,2,3,4,5,6,7,8"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figs", type=str, default=FIGS)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "C3a" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figs = args.figs.split(",")
    params = {"figs": figs, "out": args.out, "barriers": BARRIERS, "sigma": SIGMA, "string_90": STRING_90,
              "newton_90": NEWTON_90, "nucleus": NUCLEUS, "shapes": SHAPES, "finite_L": FINITE_L, "eig_N": EIG_N,
              "eig_mu": EIG_MU, "eig_full": EIG_FULL, "stairs": STAIRS, "traces": TRACES, "edge": EDGE,
              "connect": CONNECT, "coexistence": COEX, "a_rot": A_ROT, "window": WINDOW}
    style()
    with dipgpe.runs.Run("c3a_figures", params) as run:       # figures and the numbers they show, recorded
        out = [run.dir, args.out]
        res = {}
        if "1" in figs:
            fig1(out)
        if "2" in figs:
            res["shape_contrast_fwhm_um_droplets"] = fig2(out)
            print("shape (contrast, FWHM um, droplets):", res["shape_contrast_fwhm_um_droplets"])
        if "3" in figs:
            values, ac = fig3(out)
            res.update(barriers={k: round(v, 4) for k, v in values.items()}, a_c=round(ac, 4))
            print("barriers:", res["barriers"], "a_c:", res["a_c"])
        if "4" in figs:
            res["roton_end_eps_Eb_C_fwhm_EbwA2"] = fig4(out)
            print("roton end (eps: E/eps^1.5, C/eps^0.5, FWHM eps^0.5, E w/C^2):", res["roton_end_eps_Eb_C_fwhm_EbwA2"])
        if "5" in figs:
            kappa, ratios, dns = fig5(out)
            res.update(kappa_per_um_per_hbar_omega=round(kappa, 2), two_length_ratios_32_64_128=ratios,
                       delta_N=dns)
            print(f"kappa {kappa:.1f} per um per hbar omega; [E(L) - E(2L)] / prediction for 32-64, 64-128 cells:",
                  ratios)
        if "6" in figs:
            fig6(out)
        if "7" in figs:
            res["edge_drive_deficit"] = fig7(out)
            res["sigma_hbar_omega"] = {a: round(sigma(r), 3) for a, r in SIGMA}
            print("edge (a_s: drive, 2 sigma - E_b):", res["edge_drive_deficit"])
        if "8" in figs:
            res["descents_last_iteration_energy_released"] = fig8(out)
            print("descents (a_s side: last iteration, energy released):", res["descents_last_iteration_energy_released"])
        if set(figs) == set(FIGS.split(",")):
            write_sources(out, run.dir)
        else:
            print("SOURCES.txt not rewritten: only some figures were drawn")
        run.result(**res)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
