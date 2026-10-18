# Paper C3a: results and issues log

Extracted from the development log of DipGPE (newest first). Run ids refer to
`runs/<id>_*` (metadata, logs and JSON results in this repository; the fields of
the 22 critical nuclei of Table II are in the data archive, see README). Short
forms such as -045028 share the date of the id before them. Some records name
the former script `tube_bdg.py roton` (now `tube_roton.py`) and the former
package name `qs`; the code is otherwise identical.

## 2026-10-18 — C3a figures: numbers derived from the C3-2 runs

`uv run python benchmarks/c3a_figures.py` (run 20261018-140939; reads the run
records below, no simulation). Fixed μ, 32 cells (64 at 89.86–89.88 a0).
The journal-style figures of the manuscript (printed width 510 pt, 8 pt STIX)
are from run 20261018-180445 (Fig. 7(a): schematic of the staircase at 90.272
with the computed line densities, revised after the Codex review, ISSUES C3-22;
Fig. 7(c): the σ inset no longer overlaps its tick labels; Fig. 8(a): typical
case only; earlier 20261018-172653, -171201, -161440 and -154548),
whose result.json is identical. Barriers between
successive pinned states at 90.272 (from the stationary energies): M1 to S1
0.022, M1 to S2 0.144 ħω⊥ (0.16 and 1.0 nK), against E_b = 91.416 from the
uniform state:

- Shape (FWHM of the one-cell contrast, `tube_saddle.nucleus`): 47.9 µm at
  89.86, 13.8 at 90.00, 11.2 from 90.05 to 90.255 (3–5 droplets), 13.8 at
  90.265, 15.6 at 90.273; peak contrast 0.088, 0.329, 0.377–0.635, 0.671, 0.694.
- Roton end, ε = a_s − a_c with a_c = 89.8485 (E_b^(2/3) through the three
  64-cell points): for ε ≤ 0.05 a0, E_b/ε^(3/2) = 233 ħω a0^(−3/2) within 0.5%,
  contrast/ε^(1/2) = 0.821–0.824, FWHM·ε^(1/2) = 4.98–5.22 µm; E_b·FWHM/contrast²,
  which does not use a_c, 1714–1799 ħω µm (±2.4%); E_b/ε^(3/2) = 284 at 90.20.
- Finite tube without a fixed-μ reference, [E_b(L) − E_b(2L)]/[ΔN²/(4κL)] with
  κ = 929.4/µm per ħω (uniform branch of run 20261008-123945): 0.994, 1.012,
  1.021, 1.049, 1.150 (L = 32 cells) and 1.005, 1.004, 1.009, 1.022, 1.058
  (64 cells) at 89.90, 90.05, 90.20, 90.25, 90.26 (ISSUES C3-17).
- Edge: drive Δω = Δn_coex |dn_coex/da_s| (a_e − a_s)/κ = 82.9 (a_e − a_s)
  ħω/µm per a0 (run 20261008-123945); deficit 2σ − E_b = 16.86, 13.75, 10.53,
  6.96, 2.98, 1.25, 0.23 at 90.25, 90.255, 90.26, 90.265, 90.27, 90.272, 90.273
  (Δω = 1.99, 1.58, 1.16, 0.75, 0.33, 0.17, 0.086); σ = 47.79 (first written 47.80: 47.7947 rounded twice), 47.37, 46.81,
  46.20 ħω at 90.25, 90.256, 90.265, 90.274.

## 2026-10-18 — C3-2 (e): fixed-μ saddles (B) and the window at 32 cells (A)

B: Newton at fixed μ = μ_U(6250/µm) (`tube_saddle.py refine --ensemble grand`,
complex128, even sector); barrier ΔΩ = Ω_saddle − Ω_uniform with Ω = E − μN;
ΔN = atoms the nucleus adds. Gate against the fixed-N (A) values extrapolated
to L → ∞ (B runs 20261017-234923, -235005, -235035, -235057, -235127, -235257;
A 64/128 cells at 90.25, 90.26: 20261017-235519, -235618, -235743,
20261018-000133):

| a_s (a0) | centre | B, 32 cells | B, 64 cells | A, L → ∞ | ΔN | E_32 − B | ΔN²/(2κL_32) |
|---|---|---|---|---|---|---|---|
| 89.90 | gap | 2.7393 | | 2.7396 | −82.1 | 0.0414 | 0.0411 |
| 90.05 | droplet | 22.6129 | | 22.6123 | −175.0 | 0.1873 | 0.1865 |
| 90.20 | droplet | 59.2092 | 59.2082 | 59.2072 | −288.4 | 0.5122 | 0.5067 |
| 90.25 | gap | 78.7264 | | 78.7248 | −381.7 | 0.9130 | 0.8872 |
| 90.26 | gap | 83.7218 (phase condition; 83.688 stalled) | | 83.725 | −431.3 | 1.237 | 1.133 |

(ħω; A, L → ∞: 2E_128 − E_64 up to 90.20, a fit E∞ + a/L + b/L² to 32/64/128
cells at 90.25, 90.26; κ = dn/dμ = 929/µm per ħω of the uniform tube, run
20261008-123945.)

- B agrees with the fixed-N extrapolation within 1e-4 relative (3e-5 from
  90.05 to 90.25) and does not depend on the tube length (32 against 64
  cells: 2e-5).
  The 1/L term of the fixed-N barrier is ΔN²/(2κL) within 1–3% (C3-3, OURS;
  the remainder is the next order and grows with ΔN).
- At 90.26 the even-sector Newton stalled (83.688 at residual 5e-5, ISSUES
  C3-14); with a phase condition it converges to 83.7218, 3e-5 below the A fit,
  whose length differences no longer halve (ratio 2.17 instead of 2).

A, 32 cells, Morse index by parity sectors (runs 20261017-232050 to -234030;
summary 20261017-234256):

| a_s (a0) | gap-centred: E_b, index (odd eigenvalue) | droplet-centred: E_b, index (odd eigenvalue) |
|---|---|---|
| 90.20 | 59.878, – | 59.721, 1 (+3.5e-3) |
| 90.24 | 75.059, 1 (+0.7e-3) | 75.135, 2 (−1.8e-3) |
| 90.245 | 77.289, 1 (+2.1e-3) | 77.459, 2 (−3.3e-3) |
| 90.25 | 79.639, 1 (+3.8e-3) | 79.913, 2 (−4.6e-3) |
| 90.255 | 82.153, 1 (+5.4e-3) | 82.513, 2 (−5.3e-3) |
| 90.26 | 84.959, 1 (+3.5e-3) | 85.285, 2 (−4.9e-3) |
| 90.265 | 88.371, 2 (−4.8e-3) | 88.272, 2 (−2.4e-3) |

- The phases exchange between 90.20 and 90.24: from 90.24 on, the gap-centred
  four-droplet nucleus is the index-1 saddle (B: index 1 at 90.25 and 90.26).
- At fixed N the lowest even eigenvalue of the gap-centred branch approaches
  zero towards 90.26 (A: −0.029, −0.025, −0.019, −0.012, −0.004 at
  90.24–90.26), and at 90.265 the 32-cell nucleus has moved to a longer state
  (both symmetric states index 2, gap-centred FWHM 11.2 → 14.1 µm, background
  +6.4/µm). At fixed μ there is no fold there: the four-droplet nucleus exists
  at 90.265 with breathing eigenvalue −0.0049 (asymmetric search below).
- σ at the window edge (`tube_localized.py`, 32 cells, mean density at the
  middle of the coexisting pair): 46.81 ħω at 90.265 (run 20261017-231702) and
  46.20 ħω at 90.274 (run 20261017-231854), against 47.3–47.8 at 90.25–90.256:
  σ falls as the two phases become alike (coexisting density gap 57, 53, 49/µm
  and crystal contrast 0.736, 0.718, 0.700 at 90.15, 90.256, 90.35). The
  thin-wall limit at the edge is 2σ = 92.4 ħω.

Staircase test (B, 32 cells): seeds are the 90.26 nuclei lengthened by 0, 1
and 2 cells on each side (`--grow`), continued to 90.27 (runs 20261018-002122
to -004937; summary 20261018-005406). Index in brackets: the state is not
stationary (Newton stalled).

| a_s (a0) | seed | ΔΩ (ħω) | index | residual | FWHM (µm) | droplets |
|---|---|---|---|---|---|---|
| 90.265 | gap, 4 | 86.667 | (1) | 6e-5 | 13.8 | 4 |
| 90.265 | droplet, 5 | 86.913 | 2 | 5e-11 | 11.5 | 5 |
| 90.265 | gap, 6 | 83.044 | (0) | 1e-3 | 19.3 | 6 |
| 90.265 | droplet, 7 | 83.849 | (2) | 7e-4 | 17.0 | 7 |
| 90.265 | gap, 8 | 76.209 | (0) | 2e-3 | 27.4 | 10 |
| 90.27 | gap, 4 | 90.047 | (2) | 1e-4 | 13.8 | 4 |
| 90.27 | droplet, 5 | 89.969 | 2 | 7e-11 | 11.5 | 5 |
| 90.27 | gap, 6 | 88.809 | (2) | 2e-4 | 19.3 | 6 |
| 90.27 | droplet, 7 | 89.453 | (2) | 2e-4 | 17.0 | 5 |
| 90.27 | gap, 8 | 85.277 | (0) | 2e-4 | 28.1 | 10 |

- In the even sector Newton converges only for the five-droplet
  droplet-centred nucleus, with index ≥ 2 (breathing and sliding unstable; one
  odd mode computed). It stalls for the four-droplet gap-centred seed, although
  that state exists (below), and for the longer seeds, whose fronts move
  outward; these configurations have lower Ω than the five-droplet
  state (by 3–11 ħω at 90.265, 0.5–5 ħω at 90.27). That does not establish
  depinning: on a tilted washboard Ω falls from period to period even with
  pinned minima in between. A clean test is constrained continuation of one
  front over one period (review by Codex).

Asymmetric search (`tube_saddle.py asym`, `dipgpe.mep.break_symmetry`; fixed
μ, 32 cells, complex128): from the five-droplet index-2 state, the amplitude a
along its odd unstable mode is held by a constraint (translation fixed by a
phase condition) until the constraint's multiplier β(a) changes sign, then
Newton without it; index from the three lowest Hessian eigenvalues in the full
space (translation deflated, relative residuals ≤ 0.05). Runs 20261018-023307
(90.265), -024346 (90.27; the first attempt -024049 stopped when one constrained
solve past the root did not converge; such solves are now retried halfway).

| a_s (a0) | saddle | E_b = ΔΩ (ħω) | ΔN | eigenvalues (full space) | residual | 2σ − E_b |
|---|---|---|---|---|---|---|
| 90.265 | gap-centred, 4 droplets (centred at +1.36 µm, mirror-symmetric to 2e-7) | 86.6680 | −498.3 | −0.0049, +0.0010, +0.044 | 7e-11 | 6.96 |
| 90.27 | asymmetric: no mirror centre (best 1e-2), next to the five-droplet state | 89.9624 | −506.6 | −0.0080, +0.0017, +0.044 | 2e-11 | 2.98 |
| 90.272 | droplet-centred, 5 droplets (symmetric, index 1 by parity sectors) | 91.2937 | −526.2 | even −0.0033, +0.043; odd +0.0022 | 2e-10 | 1.38 |

(2σ from σ interpolated linearly between 90.265 and 90.274; phase-condition
multiplier ≤ 8e-10, so both are stationary without constraint forces.)

- Both are index-1 saddles. At 90.265 the search returns the symmetric
  gap-centred nucleus, translated by half a period: the four-droplet branch
  continues past 90.26 at fixed μ, and the even-sector stalls were a solver
  failure (86.6668 at residual 6e-5, 1e-3 ħω off; at 90.26 0.034 off; ISSUES
  C3-14).
  At 90.27 the saddle is a genuinely asymmetric (rung) state, 0.007 ħω below
  the symmetric index-2 state: between 90.265 and 90.27 the gap-centred state
  loses its sliding stability (stalled odd eigenvalue +0.0021 → −0.0006), as in
  the snaking picture.
- The deficit to the thin-wall limit falls from 6.96 to 2.98 ħω as the drive
  falls from 0.75 to 0.33 ħω/µm (ratio 9.3 and 8.9 µm with the drives 0.749 and
  0.335 of run 20261018-140939; first written 6.95 and 9.0), towards E_b = 2σ = 92.4
  ħω at the edge. Not checked: that the saddles connect the uniform and grown
  states (relaxation along both signs of the unstable mode).

Unified fixed-μ dataset (B, 32 cells, complex128; seeds: the index-1 fixed-N
saddle of each a_s; runs 20261018-035426 to -040807, earlier B runs as above).
Index by parity sectors at fixed μ where computed; 90.26 refined with a phase
condition instead of the even sector, index in the full space.

| a_s (a0) | centre | E_b = ΔΩ (ħω) | ΔN | index (fixed μ) |
|---|---|---|---|---|
| 89.86 | gap | 0.2801 (64 cells: 0.2889) | −39.4 (−38.5) | |
| 89.87 | gap | 0.7317 | −53.0 | |
| 89.88 | gap | 1.3032 | −64.0 | 1 + (odd ≈ 0) |
| 89.90 | gap | 2.7393 | −82.1 | |
| 89.95 | gap | 7.7082 | −117.2 | |
| 90.00 | gap | 14.3608 | −146.8 | |
| 90.05 | droplet | 22.6129 | −175.0 | |
| 90.10 | droplet | 32.5659 | −204.7 | 1 |
| 90.15 | droplet | 44.5244 | −239.5 | |
| 90.20 | droplet | 59.2092 | −288.4 | 1 |
| 90.21 | droplet | 62.6109 | −302.1 | |
| 90.22 | droplet | 66.2274 | −318.4 | |
| 90.23 | droplet | 70.1056 | −338.3 | 1 |
| 90.24 | gap | 74.2795 | −354.3 | |
| 90.245 | gap | 76.4513 | −366.7 | |
| 90.25 | gap | 78.7264 | −381.7 | 1 |
| 90.255 | gap | 81.1315 | −401.1 | 1 |
| 90.26 | gap (phase condition) | 83.7218 | −431.3 | 1 (full space) |
| 90.265 | gap (asymmetric search) | 86.6680 | −498.3 | 1 (full space) |
| 90.27 | asymmetric | 89.9624 | −506.6 | 1 (full space) |
| 90.272 | droplet | 91.2937 | −526.2 | 1 |

- All converged (residual ≤ 1e-9). The fixed-N 32-cell barriers lie above these
  by ΔN²/(2κL) within 1–3% (e.g. 90.24: 0.780 against 0.765).
- 89.86: the nucleus (FWHM ≈ 50 µm) does not fit the 88 µm tube; 64 cells give
  +3%; 89.87 and 89.88 at 64 cells: 0.7354 (+0.5%) and 1.3049 (+0.1%) (runs
  20261018-045759, -045934). At 89.90 (FWHM 22 µm) 32 cells agree with the
  L → ∞ fixed-N value to 1e-4.
- E_b^(2/3) is linear near the roton end (64-cell values at 89.86–89.88:
  successive slopes 37.8, 37.9 per a0, and 38.2 on to 89.90; first written
  37.7): a_c = 89.848 (BdG a_rot* =
  89.845–89.847).
- 90.26: with the phase condition Newton converges (residual 6e-11) to 83.7218,
  3e-5 below the three-length fixed-N fit 83.7247; the even-sector value 83.688
  (stalled at 5e-5) was 0.034 too low.

Connectivity (`tube_saddle.py connect`, `dipgpe.mep.descend`: accelerated
descent of Ω at fixed μ with backtracking, from the saddle ± 0.05 × nucleus
norm along its unstable mode; even sector for symmetric saddles, phase
condition otherwise; complex128, 4000 iterations per side, ≈ 22 ms each). The
shrinking side counts as reached at contrast < 0.02 and |Ω − Ω_U| < 1e-3 E_b;
the growing side when the nucleus is 3 cells longer with Ω below the saddle.
Runs 20261018-042650 (90.20), -044820, -045028, -045110, -045311, -045542,
-050130 (90.27, growing side 20000 iterations).

| a_s (a0) | saddle | shrinking side | growing side |
|---|---|---|---|
| 89.88 | gap | contrast 0.012, Ω − Ω_U = 0.030 after 4000 (halving every 500) | whole tube crystallizes (Ω − Ω_U = −1277) |
| 90.10 | droplet | uniform (950 it.) | grows (Ω − Ω_U = −167, 800 it.) |
| 90.20 | droplet | uniform (850 it.) | grows (Ω − Ω_U = −5.7, 1250 it.) |
| 90.26 | gap | uniform (1400 it.) | grows slowly: FWHM 11.5 → 19.3 µm, Ω 6.3 below the saddle after 4000 |
| 90.265 | gap | uniform (2750 it.) | grows slowly: FWHM 14.0 → 19.2 µm, Ω 3.5 below the saddle |
| 90.27 | asymmetric | uniform (2100 it.) | bottleneck 0.33 below the saddle (see below) |
| 90.272 | droplet | uniform (2300 it.) | pinned localized minimum 0.022 below the saddle |

- Every saddle connects the uniform superfluid with a growing nucleus; near the
  edge the growth is slow (drive 0.7–0.3 ħω⊥/µm).
- 90.27: after 20000 iterations Ω is 0.33 below the saddle (FWHM 16.4 µm,
  ΔN = −649) with the residual at 1.7e-6 and Ω still falling by ~4e-7 per
  iteration. Newton finds no stationary state there (run 20261018-051634,
  stalled at 1.7e-6), and the softest Hessian eigenvalue changes sign between
  neighbouring states (−3e-4 here, +6e-5 at a nearby state, run
  20261018-050914): the ghost of a pinned state near the depinning threshold of
  the front, where the Peierls barrier is about Δω d / π ≈ 0.3 ħω⊥. Along the
  path followed nothing rises above the saddle, so E_b(90.27) = 89.962 holds
  as far as the descent went (20000 iterations).
- 90.272 (runs 20261018-052437, -053925, -054330): the five-droplet
  droplet-centred nucleus is again sliding-stable (snaking) and is the index-1
  saddle itself (91.2937). Its growing side converges to a pinned localized
  minimum 0.022 below it (Newton 2e-11, full-space index 0: +0.0024, +0.0057,
  +0.039; ΔN = −575): the front depins between 90.272 and 90.27, so the
  Peierls barrier is ΔP = Δω_c d / π ≈ 0.15–0.29 ħω⊥, and from 90.272 on the
  growth is a staircase. Further growth must leave that minimum over a next
  saddle. (First written 2026-10-18 as "the first saddle sets E_b" on a
  tilted-washboard argument; that does not hold next to a near-fold pair.)
  The staircase, one front at a time (`tube_saddle.py staircase`, run
  20261018-103344; each state Newton-converged and indexed in the full space):

  | state | Ω − Ω_U (ħω⊥) | ΔN | index |
  |---|---|---|---|
  | five-droplet saddle | 91.2937 | −526 | 1 |
  | pinned minimum | 91.2716 | −575 | 0 |
  | next saddle (right front advances) | 91.4159 | −654 | 1 (−0.0053, +0.0045, +0.032) |
  | next minimum | 90.5493 | −904 | 0 (+0.0041, +0.0043, +0.019) |

  So at 90.272 E_b = 91.4159 along this path, 0.122 above the five-droplet
  saddle, unless a later saddle is higher: the next minimum is 0.87 below it,
  against a climb of 0.14 from the previous minimum, so a higher one is
  unlikely (not computed). Deficit to 2σ: 1.25 (ratio to the drive 7.4 µm, first written 7.5; the
  sequence 6.96 → 2.98 → 1.25 for drive 0.75 → 0.33 → 0.17 ħω⊥/µm).
  Stepping on from the minimum 90.5493 (run 20261018-105238) jumped to other
  branches (ISSUES C3-16) but found only lower index-1 states of larger nuclei
  (90.158 at ΔN = −1089, 89.736 at −1234): no saddle above 91.4159 was met.
  At 90.273 the even-sector Newton from the 90.272 state stalls (residual 1e-4,
  Ω − Ω_U ≈ 92.00) with the breathing eigenvalue at −3e-4: the five-droplet
  saddle meets its pinned minimum in a fold near 90.273 (runs 20261018-053524,
  -053733). The asymmetric search at 90.272 found no sign change, as expected
  for an index-1 symmetric state (run 20261018-052617).
- 90.273 (runs 20261018-110707, -110931, -111156, -111328): the 90.272 second
  saddle and minimum continue (Newton 5e-13 and 1e-13): saddle 92.3055
  (ΔN = −654, index 1: −0.0057, +0.0049, +0.032) and minimum 91.8212
  (ΔN = −887, index 0). From the saddle the shrinking side slows at
  Ω − Ω_U ≈ 92.03 within 4000 iterations (residual 1e-5 and falling): the
  bottleneck left by the five-droplet pair, below the saddle. Stepping on from
  the minimum again jumped branches and found only a lower index-1 state
  (91.734 at ΔN = −1089). E_b(90.273) = 92.3055 along this path; 2σ(90.273)
  ≈ 92.53 (σ interpolated), deficit ≈ 0.23 ± 0.3 (first written 92.54 and ±0.2;
  the 32-cell, 64-cell and reference-free σ differ by up to 0.13, C3a Appendix D,
  so 2σ is uncertain by about ±0.3). Near the edge the
  pinning can add up to the Peierls barrier to 2σ.

## 2026-10-17 — C3-2 (d): towards the coexistence window (32 cells)

Droplet-centred saddle continued from 90.20 a0 (Newton, complex128, even
sector; index by parity sectors). Runs 20261017-223225, -223326, -223431,
-223553 (32 cells), -223713 (64 cells, 90.22); summary run 20261017-223830.

| a_s (a0) | E_b (ħω) | index (odd sliding eigenvalue) | contrast | droplets |
|---|---|---|---|---|
| 90.20 | 59.721 | 1 (+3.5e-3) | 0.543 | 3 |
| 90.21 | 63.174 | 1 (+3.2e-3) | 0.557 | 3 |
| 90.22 | 66.854 (64 cells 66.537) | 1 (+2.4e-3) | 0.572 | 3 |
| 90.23 | 70.816 | 1 (+0.8e-3) | 0.589 | 3 |
| 90.24 | 75.135 | 2 (−1.8e-3) | 0.607 | 3 |

- No thin-wall regime before the coexistence window (90.237–90.274 a0): the
  nucleus stays a compact patch of three droplets (FWHM 11.2 µm, contrast at 0,
  1, 2, 3 cells from the centre 0.61, 0.57, 0.36, 0.15 at 90.24), only deeper
  (centre contrast towards the bulk crystal's 0.79). Its length is that of
  about one interface (~9 µm), so no flat top between two walls forms; E_b
  rises with a growing slope (3.5, 3.7, 4.0, 4.3 ħω per 0.01 a0) and is 75 ħω
  at 90.24, below 2σ ≈ 95 ħω. The 2σ limit applies only where the driving force
  vanishes (at fixed n the uniform-side edge 90.274 a0, where the superfluid
  at n is itself at coexistence; ISSUES C3-13) and is approached logarithmically
  slowly (deficit ∝ Δf ln(1/Δf)); it is not tested here.
- Inside the window (90.24) the sliding eigenvalue of the droplet-centred
  saddle changes sign: the phases exchange sliding stability again (snaking),
  part of C3-2 (e).

## 2026-10-17 — C3-2 (c), (f): tube length and grid convergence of the barrier

Newton (complex128, even sector) from the 32-cell saddles embedded in longer
tubes (uniform state added on both sides) or interpolated to other grids;
no Hessian at ≥ 64 cells (ISSUES C3-11). Summary run 20261017-221429.

| a_s (a0) | centre | 32 cells | 64 cells | 128 cells | 2·E_128 − E_64 |
|---|---|---|---|---|---|
| 89.90 | gap | 2.7807 | 2.7603 | 2.7499 | 2.7396 |
| 90.05 | droplet | 22.8002 | 22.7059 | 22.6591 | 22.6123 |
| 90.20 | droplet | 59.7214 | 59.4628 | 59.3350 | 59.2072 |

(E_b in ħω; runs 32: 20261017-202122, -205917, -210726; 64: -215133,
-220616, -220704; 128: -221037, -221204, -221259.)

- Fixed N: E_b decreases with the tube length exactly as 1/L (successive
  differences in the ratio 1.98, 2.02, 2.02 when L doubles), the depletion
  correction of the canonical ensemble; the last column is the L → ∞ value.
  The 32-cell (88 µm) barriers lie 0.8–1.5% above it.
- dx = 0.1875 l (nz 23 per cell) at 90.05 a0: 22.7987 ħω, −0.007% (run
  20261017-220746): converged in dx.
- L_perp = 32 l at 90.05 a0: 23.3288 ħω, +2.3% (run 20261017-220811). The tube
  cut-off radius moves the BdG roton spinodal from 89.845 to 89.842 a0 (runs
  20261005-143507, -185459); with E_b ∝ ε^(3/2) this predicts 22.80 ×
  (0.208/0.205)^(3/2) = 23.30 ħω, within 0.1%. Barriers are therefore quoted
  against ε = a_s − a_rot* of the same grid.

## 2026-10-17 — C3-2 (a), (b): tube saddles from the roton end to 90.20 a0 (32 cells, preliminary)

6250/µm, 32 cells (88 µm), dx = 0.25 l, L_perp = 24 l, complex128, even sector;
Newton by continuation in a_s, residuals ≤ 6e-10. Two phases of the nucleus: a
gap or a droplet at its centre (z = 0). Gap: continuation from the 90.0 a0
saddle (runs 20261017-201116 to -203720); droplet: string at 90.05 a0 with a
half-cell-shifted seed (pilots 20261017-204602, -205029), then Newton and
continuation (runs 20261017-205917 to -210726). Summary: `uv run python
benchmarks/tube_saddle.py summary --runs <the eleven refine runs>` (run
20261017-210932). Not yet converged in L, dx, L_perp.

| a_s (a0) | centre | E_b (ħω) | Morse index (even + odd) | FWHM (µm) | contrast |
|---|---|---|---|---|---|
| 89.86 | gap | 0.2898 | 1 + (odd ≈ 0) | 50.8 | 0.088 |
| 89.87 | gap | 0.7491 | 1 + (odd ≈ 0) | 34.9 | 0.122 |
| 89.88 | gap | 1.3284 | 1 + (odd ≈ 0) | 29.4 | 0.148 |
| 89.90 | gap | 2.7807 | 1 + (odd ≈ 0) | 21.6 | 0.189 |
| 89.95 | gap | 7.7923 | 1 + (odd ≈ 0) | 16.4 | 0.267 |
| 90.00 | gap | 14.4926 | 1 + 1 (−3.4e-6) | 13.8 | 0.331 |
| 90.05 | gap | 22.8018 | 1 + 1 (−1.0e-4) | 11.2 | 0.389 |
| 90.05 | droplet | 22.8002 | 1 + 0 (+1.0e-4) | 11.2 | 0.379 |
| 90.10 | droplet | 32.8225 | 1 + 0 (+6.3e-4) | 11.2 | 0.430 |
| 90.15 | droplet | 44.8762 | 1 + 0 (+2.0e-3) | 11.2 | 0.483 |
| 90.20 | droplet | 59.7214 | 1 + 0 (+3.5e-3) | 11.2 | 0.543 |

(odd ≈ 0: the lowest odd eigenvalue is within its LOBPCG error of zero.)

- Roton end, amplitude-equation scaling with ε = a_s − a_c: E_b^(2/3) is linear
  in a_s (first four points within 0.1%; a_c = 89.849 a0; local exponents
  1.47–1.58 up to 90.05), FWHM·ε^(1/2) = 5.1 ± 0.2 µm and contrast/ε^(1/2) =
  0.84–0.87 over 89.86–90.05. The 3/2 exponent is our derivation (C3-3), here
  supported numerically. a_c agrees within 0.002–0.004 a0 with the BdG roton
  spinodal of the uniform tube on the same grid, a_rot* = 89.845 (run
  20261005-143507; 89.847 with a finer q window, run 20261006-145707; 89.845 at
  dx = 0.1875 and 89.842 at L_perp = 32): the barrier vanishes at the roton
  spinodal. (C1's 89.83 is the static estimate; RESULTS 2026-10-05.) Above 90.05 E_b rises
  faster than ε^(3/2) (90.20: E_b^(2/3) 13% above the line), towards the thin
  wall limit 2σ ≈ 95 ħω.
- The two phases are mirror images in their sliding stability (Burke &
  Knobloch 2006): at 90.05 the odd sliding eigenvalue is −1.0e-4 (gap) and
  +1.0e-4 (droplet), and the droplet-centred nucleus is lower by 0.0016 ħω; it
  is the index-1 saddle from 90.05 to 90.20. Near a_c the sliding mode is ~0 and
  the phases are degenerate within the precision.

## 2026-10-17 — C3-G: exact gates for the saddle tools

`uv run pytest tests/test_c3_gates.py` (6 tests, CUDA, 29 s; complex128).

- G2, free particle on a periodic box (L = 2π): at cos(jx) the tangent-space
  Hessian has the eigenvalues (n² − j²)/2 of the other eigenvectors, with
  sin(jx) a zero mode: [−0.5, 0, 1.5, 1.5] at j = 1 and [−2, −1.5, −1.5, 0, 2.5,
  2.5] at j = 2, to 1e-9; deflating sin(jx) leaves index 2j − 1. Newton from
  cos x + 0.1 (cos 2x + 1) + 0.05 sin x (translation deflated) reaches the
  eigenspace: residual 8e-14, E = µ = 1/2 to 2e-16.
- G3, repulsive GP on a ring (g = 1, L = 2π, m = 0.9, 256 points): the
  analytic states A sn(kx | m) (2j nodes) are stationary on the grid to
  residual 2e-12, with N and µ matching the closed forms to 12 digits. Newton
  from a perturbed state (residual 0.4–1.8) converges in 5 steps to residual
  1e-12 and relative distance 3e-14 from the analytic state. Morse index with
  the translation mode deflated: 1 at j = 1 (−0.0106, then +4.04) and 3 at
  j = 2 (−0.0424, −0.0213 twice, then +15.19), i.e. 2j − 1 as continued from
  g = 0; without deflation the translation mode is a zero eigenvalue.
- The Jacobi functions are computed in the test by the arithmetic-geometric
  mean (Abramowitz & Stegun 16.4, 17.6; no SciPy); K(0.9), E(0.9) match Table
  17.1 to 1e-14 and sn is checked by inverting the elliptic integral.

## 2026-10-16 — C3-1: string method and constrained Hessian (toy gates)

`dipgpe.mep.string_method` (string method on the norm sphere, real fields,
normal-part steps with the upwind tangent, combined preconditioner, climbing
image) and `dipgpe.bdg.constrained_hessian` (lowest eigenvalues of M on the
sphere's tangent space). `uv run pytest tests/test_mep.py`: 8 tests, pass on
CUDA (8.0 s) and CPU (5.2 s).

- Du & Zhang sphere toy (R³): saddle (0, 1, 0) to 1e-8, barrier 2 to 1e-12,
  path on the equator, 9 and 17 images.
- 1D harmonic trap, g = 0 (exact: saddle φ1, barrier µ1 − µ0 = 1, Hessian
  −1, 1, 2): barrier error 4e-16 (complex128) / 1e-7 (complex64); saddle
  within 6e-11 of φ1; Hessian to 1e-14 / 2e-6.
- g = 5 (dark soliton; reference = odd-parity minimization, complex128):
  barrier 0.86513581612209, error ≤ 5e-16 (complex128), 3e-7 (complex64);
  9 and 17 images agree to 1e-15; Morse index 1 at the saddle (−0.351,
  +1.915, +2.467), 0 at the ground state.
- Converged string: normal gradient ≤ 1e-10 (complex128) / 1e-6 (complex64)
  on every image for h = 0.125–0.5; the path converges at first order in the
  image spacing (largest distance between strings of 9/17, 17/33, 33/65
  images: 1.8e-2, 9.8e-3, 5.2e-3).
- `dipgpe.mep.refine_saddle` (Newton on the sphere, MINRES on the indefinite
  Hessian M, kinetic preconditioner, linear tolerance 1e-2): from a coarse
  string (climbing residual 5e-3) the dark soliton in 4 steps (residual 1e-11,
  4–7 MINRES iterations each), barrier exact to 1e-15 (complex128) / 2e-7
  (complex64); from the rough odd guess χ (residual 0.6) in 5 steps; φ1 from
  φ1 + 0.1 (φ0 + φ3) in 5 steps. 14 tests in tests/test_mep.py (CUDA 12.6 s,
  CPU 6.1 s).

## 2026-10-08 — Coexistence and localized states (Appendix D of the paper)

- **Coexistence** (`benchmarks/tube_coexistence.py`, run `20261008-123945`;
  `-123401` superseded): Maxwell construction at 6250/µm, a_s = 90.15, 90.256,
  90.35. Coexisting densities differ by 53/µm (crystal less dense); at fixed
  n = 6250/µm phase separation (thermodynamic limit) for 90.237–90.274 a0,
  width 0.037 a0, containing a* = 90.2555 (= submitted value). Identity residuals
  below 2e-5; width independent of the cell length. Convergence (runs
  `20261008-154519`, `-154742`, `-154909`): density step 10/µm and dx 0.1875 l
  change the window by < 1e-5 a0; transverse box 32 l shifts it by −0.0024 a0
  (width 0.0369).
- **Localized states** (`benchmarks/tube_localized.py`, runs `20261008-130517`,
  `-131726`, `-132115`, `-132458`): stationary crystal patches at n_M in
  superfluid at n_U (densities within 6/µm), interfaces ≈ 9 µm, σ = 47.3–47.9
  ħω⊥ per interface (bulk reference and 32/64-cell comparison agree). In the
  88 µm tube they lie above the homogeneous states; phase separation pays off
  only above ≈ 250 µm near a* (513 µm at 90.25 a0).


## Issues found and resolved

| # | Problem | Cause | Resolution | Status |
|---|---|---|---|---|
| C3-22 | Codex review (gpt-6-astra, max effort, read-only) of the text and Fig. 7 on growth near the edge: the barrier along the computed segment U to M2 was stated as if it were the global barrier; the depinning threshold was stated as absolute at finite temperature; the two escape barriers of M1 were called the barriers between successive pinned states; Fig. 7(a) had an uncomputed crest after M2 and dipped below the uniform level (spline overshoot); 'merged and disappeared' at 90.273 and 'only within' overstated unsuccessful searches | Wording and drawing went beyond the computed states | Text qualified: along the computed path, higher later saddles and lower routes not excluded; deterministic depinning, thermal motion possible below it; escape barriers of M1; at 90.273 'not recovered, consistent with a fold' (runs -053524 stalled at residual 1.1e-4, -053733 failed); 'found within 0.002 a0'; Delta P 'of the same scale'. Fig. 7(a) redrawn with zero slopes at the computed stationary points only, dashed after M2, comparator labelled conceptual (run 20261018-172653). Not adopted: softening 'approaches twice the interface tension', supported by the deficit 0.23 +- 0.3 at 90.273 and stated with the 2 sigma + Delta P limit in IV E | fixed |
| C3-21 | C3a Sec. IV F said that every saddle connects the uniform superfluid with a growing nucleus, and Fig. 8(b) labelled the energy of the saddle a descent starts from as E_b; at 90.272 the descents start from the first saddle, which leads to a pinned minimum, and E_b is set by the second, higher saddle | The 90.272 staircase was mapped after Sec. IV F and Fig. 8 were drafted | IV F and the Fig. 8 caption corrected (the dotted line is Omega_s - Omega_U, equal to E_b except at 90.272); new Fig. 7(a) shows the staircase with the computed line densities; IV E and the Discussion distinguish the barrier from the uniform state (91.416) from the barriers between pinned states (0.022, 0.144) and state that the latter are below k_BT at 5 nK | fixed |
| C3-20 | C3a Sec. II cited C1 (arXiv 2610.12511) for the coexistence window and the localized states, which are only in the unposted C1 v2; the number audit checked C1-cited numbers against C1 v2.1 rather than the cited v1 | C1 v2 was never posted to arXiv, and the audit read the newest local C1 text | Author's decision (2026-10-18): C1 not replaced; C3a presents the Maxwell construction, the window and σ itself (new Appendix D), and the audit checks C1-cited numbers in the arXiv v1 source (all present there). Writing Appendix D found two more points: σ(90.25) printed 47.80 was 47.7947 rounded twice (now 47.79), and the 32-cell, 64-cell and reference-free σ differ by up to 0.13, so 2σ is uncertain by ±0.3, not ±0.2 (text, Fig. 7, RESULTS). The audit now takes unrounded values from the figure code and fails a value on a rounding boundary | fixed |
| C3-19 | The C3a number audit (a script that checks every number of the manuscript against its source, 199 checks) found three slips in the draft: Sec. IV B gave "successive slopes 37.7, 37.9 and 38.2" for three points; Appendix C gave the 90.265 deficit as 6.95 (6.96 in Sec. IV E) and ℓ* + ξ = 9.0 µm at 90.27 | The slopes had been copied from RESULTS, where 37.77 was truncated and the third slope runs on to 89.90; 6.95 truncated 6.957; the ratio 9.0 and ξ, ℓ* came from the theory note's rounded drive 0.33 instead of 0.335 | Text corrected (37.8, 37.9 and 38.2 on to 89.90; 6.96; 8.9 µm, ξ ≈ 1.4–1.8 µm, ℓ* ≈ 7.5 µm, with the drives printed as 0.749 and 0.335 so that the ratios can be checked), RESULTS and the theory note corrected with a note (there also the 90.272 ratio, 7.5 → 7.4 µm); every number of the manuscript now has a checked source | fixed |
| C3-18 | Draft C3a Fig. 7(a): the descents from a saddle started beside it with a hook in the (ΔN, Ω) plane | A descent starts from the saddle plus 0.05 × the nucleus norm along the unstable mode, which changes N at once by 23–46 atoms; in its first 50 iterations the background refills at fixed μ (11–15 atoms) before the nucleus moves (90.272 shrinking side: −ΔN 526 → 480 → 491, then falling; run 20261018-053925) | Traces drawn from the saddle with the first 50 iterations dashed (`c3a_figures.py`) | not a bug |
| C3-17 | Draft C3a Fig. 5(b): (E_b(N, L) − ΔΩ)/(ΔN²/(2κL)) rose with the tube length at 89.90 a0 (1.007, 1.020, 1.035 at 32, 64, 128 cells), against the trend at the other a_s | The reference ΔΩ is the fixed-μ barrier at 32 cells, with its own finite-size error: ≈ 3e-4 ħω⊥ low at 89.90 (FWHM 22 µm in an 88 µm tube; the fixed-N barriers extrapolated to L → ∞ give 2.7396–2.7397 against 2.7393), while the excess at 128 cells is only 0.0106 | Fig. 5(b) compares two tube lengths instead: [E_b(L) − E_b(2L)]/[ΔN²/(4κL)] = 0.994, 1.005 at 89.90 and → 1 with L at all five a_s. The 1e-4 relative agreement of ΔΩ with the extrapolated barriers (RESULTS C3-2 (e)) stands | mitigated |
| C3-16 | Staircase steps from a minimum reached a = 0.5 × (last step) with Ω below the start although β > 0 there (runs 20261018-105238, -111328); several constrained solves did not converge | After a large drop (saddle to minimum, ΔN changes by ~250 atoms) the last step is long, and amplitudes in its units make the first constrained Newton start far from the start state: it converges to another branch, so the scan is not a continuous path | The states found are still stationary and indexed (and here all lower than the barrier); for a continuous path the amplitudes must be a small fraction of a long last step (e.g. 0.05–0.5) | mitigated |
| C3-15 | Newton from a stored fixed-μ state (the end of a descent at 90.27, residual 1.7e-6) started at residual ~1e-4 and stalled at 5e-5 (run 20261018-050914) | `start_state` renormalized every start to the uniform N, a fixed-N step applied to a fixed-μ state with ΔN = −649 | Fixed-μ starts keep their norm (`tube_saddle.py`); the rerun starts at 1.7e-6 (run 20261018-051634). Earlier fixed-μ runs started from fixed-N saddles (norm N already) or converged anyway | fixed |
| C3-14 | Newton at fixed μ in the even sector stalls at residual 5e-5 to 1e-4 at 90.26, 90.265, 90.27 a0 for the gap-centred four-droplet nucleus (runs 20261017-235257, 20261018-002122, -003736), and at up to 2e-3 for longer seeds; MINRES iterations grow from ~180 to ~500 per step | First read as a fold of the four-droplet branch (its fixed-N even eigenvalue goes to zero; the fixed-μ trend first quoted also mixed in the droplet-centred state at 90.20, found in review). Not a fold: the same state, translated by 1.36 µm, converges at 90.265 to 7e-11 with the phase-condition Newton of break_symmetry (run 20261018-023307; index 1, breathing eigenvalue −0.0049). Why the even-sector Newton stalls is not understood | Near-edge saddles by break_symmetry with a phase condition and the index in the full space (RESULTS C3-2 (e)); the stalled even-sector values were 1e-3 ħω off at 90.265 (86.6668 against 86.6680) and 0.034 off at 90.26 (83.688 against 83.7218 with a phase condition, `tube_saddle.py refine --sector none`) | mitigated |
| C3-13 | PLAN C3-2 (d) put the thin-wall limit E_b → 2σ "near a*" (90.256), and it was described in conversation at the crystal-side edge of the window (90.237) | At fixed n the nucleus grows out of an effectively infinite superfluid at μ_U(n); the bulk driving force is Δω ≈ (n_U − n_M)(μ_coex − μ_U(n)), which vanishes where the superfluid at n is itself at coexistence: the uniform-side edge 90.274 a0 (run 20261008-123945: crystal less dense by 53/µm, coexisting densities −1448/µm per a0; dμ/dn = 1.076e-3 ħω µm, so Δω ≈ 83 (90.274 − a_s) ħω/µm; first written with dμ/dn per l, 1.677e-3, corrected 2026-10-18). a* and 90.237 concern the end state, not the nucleus. In a fixed-N tube the expelled atoms (ΔN = −288 at 90.20, RESULTS C3-2 (e)) raise μ of the background, a counter-force Δn² ℓ/(κL) that exceeds Δω near the edge (32 cells: from ≈ 90.27) | The limit is tested at the uniform-side edge; at fixed N only with L → ∞ (within ≈ 0.005 a0 of the edge the depletion is ≥ 20% of the drive even at 128 cells, so the 1/L extrapolation is unreliable there) or in the grand-canonical ensemble (fixed μ = μ_U(n)) | not a bug |
| C3-12 | 128-cell runs ran out of GPU memory before the first Newton step (runs 20261017-220610, -220657, -220739) | Building the tube-cutoff dipolar kernel for the linear response (full kz grid): the quadrature span 40/q_min and the number of |kz| both grow with the tube length, so the (nodes × |kz|) Bessel tables grow as L² (~1 GB each at 128 cells) | `_tube_outer` builds the tables per k block and q block; kernel identical (maximum difference 0.0 at 32 and 64 cells), 2 s and 2.2 GiB at 128 cells; 128-cell Newton peak 9–10 GiB | fixed |
| C3-11 | A 64-cell run made the PC unusable: GPU memory full (15.9 of 16.3 GB) and the Python process at 30 GB of private memory, 0.2–0.5 GB of RAM free (run 20261017-215133); a second time an orphaned batch step ran next to a new job | The parity-sector Hessian (LOBPCG) at 64 cells needs ~12–14 GB (blocks of up to 12 complex128 fields of 160 MB, stored as X, AX, P, AP, W, AW and again concatenated as S, AS, plus the M temporaries; measured 11.6 GiB after three Rayleigh-Ritz steps, then an allocation of 1.8 GiB); run after Newton in the same process, cache fragmentation pushed it over the card, and the Windows driver spilled to system memory instead of failing. Not the cuFFT plan cache (6 plans). Stopping the batch shell left its child jobs running | `tube_saddle.py` caps GPU memory (`torch.cuda.set_per_process_memory_fraction`, default 0.85: an allocation beyond it raises OutOfMemoryError, checked) and empties the cache before the Hessian; no Hessian at ≥ 64 cells for now (index from 32 cells); batches are stopped with their process tree; one GPU job at a time. A leaner LOBPCG (no S/AS copies, real fields) would halve the memory if long-tube indices are needed | mitigated |
| C3-10 | The tube saddles carried an odd (parity-breaking) part of 0.6–2.5e-3 of the norm, and Newton stalled at residuals 1e-8 to 1e-6 (runs 20261016-182258 to 20261017-200206) | Round-off in the long complex64 string grew along the odd sliding mode (droplets against the envelope), whose eigenvalue is within ~1e-4 of zero; Newton without the sector then stops near an asymmetric point | `string_method` and `refine_saddle` take a symmetry `sector`; the tube runs use the even sector: residuals 1e-10 to 2e-11 in 2–3 steps, barriers unchanged to 1e-6 (runs 20261017-201116 to -203720). Finding: in the even (gap-centred) saddle the odd sliding eigenvalue is ~0 near a_rot* and negative at 90.00 and 90.05 (−3.4e-6, −1.0e-4): index 2 there; the droplet-centred phase is to be computed (C3-2 (b)) | fixed |
| C3-9 | Newton in the 3D tube stalls at residual ~1e-8 (32 cells) or 2e-9 (64 cells), above the 1e-9 default, and each further step took ~700 MINRES iterations (runs 20261017-181657, -181954) | Not identified; likely MINRES stopping in the preconditioned norm, so the true residual of the step is not reduced below the floor | `refine_saddle` stops when a step lowers the residual by less than half (info["stalled"]); the energy is converged to second order in the residual | mitigated |
| C3-8 | 3D Hessian too slow: 600 LOBPCG iterations per parity sector at 32 cells (9 min) for residual 1e-3, and 2 s per iteration at 64 cells (4.4x the 32-cell cost); a 64-cell run was killed by its time limit (run 20261017-181954) | B = 1 blocks were stored as copies (memory near the 16 GB limit); the 1e-3 tolerance is stricter than the Morse index needs | B = 1 is not stored (32 cells 0.46 -> 0.32 s per iteration; 64 cells peak 11.8 GiB, still 1.7 s); for the index the tolerance 0.3 certifies each sign (error < |λ|): 4.3 min at 32 cells (run 20261017-184545); at 64 cells the odd sliding mode (+1.3e-4) stays uncertified after 400 iterations (run 20261017-185005) | mitigated |
| C3-7 | `CUDA error: unknown error` during the Hessian at 89.95 a0 (32 cells, run 20261017-181657), raised at a synchronization in LOBPCG | Unknown (asynchronous; driver or GPU fault); the same saddle and code path ran cleanly in run 20261017-184545 | Watch for recurrence | open (not reproduced) |
| C3-6 | The Morse index of the 3D saddle was not resolved: the second Hessian eigenvalue stayed at −9e-7 with residual 47 after 300 LOBPCG iterations (run 20261016-182258) | A near-zero odd mode (droplets sliding against the patch envelope, +1.2e-4) next to the deflated translation | Parity sectors (`constrained_hessian(sector=...)`): even −0.0572, +0.0242; odd +1.2e-4, +0.0253 (run 20261016-182658): index 1 | fixed |
| C3-5 | 3D tube (6250/µm, 90.0 a0, 32 cells, 96×96×544, complex64): the string with step 0.5 never settled (image energies jumped between 5e3 and 8e4 ħω, run 20261016-171253); with step 0.2 it is stable but slow: the climbing image's residual falls by e every ~3000 iterations (run 20261016-173115), i.e. ~35,000 iterations (~3.5 h at 9 images, complex128) to the complex128 tolerance; complex64 energies scatter by ~0.5 ħω (E ≈ 4.6e6 ħω) | Droplet densities make the Hessian ~3x stiffer than the preconditioner assumes (stability); fixed-step gradient descent converges at the rate of the softest mode at the saddle | Step 0.2 for the tube; `mep.refine_saddle` (Newton, MINRES) finishes the saddle: at 90.0 a0 from the climbing image (residual 1.5e-4) in 4 steps, 20 s, complex128 (run 20261016-182258); it stalls at residual ~1e-8 (above the 1e-9 default; E is converged to second order) | fixed |
| C3-4 | `constrained_hessian` in complex64 did not converge from some starts: on the CPU Ritz values ran to −6e13 (true lowest −0.35); on CUDA the 1D-trap test failed for one random seed and passed for others | LOBPCG in float32: the error of A S (~eps ||A|| ~ 4e-5 here) is amplified by the inverse of small eigenvalues of S^H S kept in the Rayleigh-Ritz step, and the outcome depends on rounding | The Bogoliubov operators act in double precision for any field precision (complex64 state promoted, complex64 dipolar kernels kept) and LOBPCG/MINRES blocks are complex128; the 1D tests converge for every seed tried | fixed |
| C3-3 | The simplified string (full step, then redistribution) keeps a normal gradient proportional to the step h after convergence: 1D trap (g = 5, 17 images) 5.0e-3, 2.7e-3, 1.4e-3 for h = 0.5, 0.25, 0.125 | Splitting error: the full step moves images along the curved path, and the redistribution puts them back on a slightly shifted curve | Normal-part steps with the upwind tangent (C3-2): normal gradient 1e-10 (complex128) / 1e-6 (complex64) on every image, independent of h; barrier unchanged (to 1e-15); the path converges at first order in the image spacing | fixed |
| C3-2 | Moving only the normal part of the gradient (Du & Zhang's general string dynamics) with the spline tangent gave zig-zag strings that never converged (sphere toy, 17 images: normal gradient stuck at 0.91, images at z = ±0.2) | The tilt of a centred tangent couples the tangential gradient to the zig-zag mode like advection; an explicit step with a centred difference is unstable for it | Upwind tangent (Henkelman & Jonsson 2000, difference to the higher-energy neighbour): sphere and 1D toys converge in the same iteration counts as the simplified method | fixed |
| C3-1 | Fixed-step string in a harmonic trap did not converge (change per step stayed 1.7) with the kinetic preconditioner (abs(mu) + k^2/2)^-1 | At the box edge V = 32 against mu ~ 1, so P(H - mu) ~ 20 and h P(H - mu) > 2 for h = 0.5 | Combined preconditioner of `minimize.py` with P_V normalized, abs(mu) / (abs(mu) + U - min U), so P ~ 1/abs(mu) where U is low and ~ 1/U where it is high | fixed |
