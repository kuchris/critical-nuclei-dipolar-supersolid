"""Gates for the string method (dipgpe.mep) and the constrained Hessian: saddle
points with known positions, barriers and Morse indices."""

import math

import pytest
import torch

import dipgpe
from dipgpe.bdg import constrained_hessian
from dipgpe.mep import FieldSpace, break_symmetry, descend, path_through, refine_saddle, string_method
from util import C64, C128, DEVICE


class Sphere:
    """R^3 with E = (x^2 - 1)^2 + y^2 + 2 z^2 on the unit sphere (Du & Zhang 2009,
    Sec. 5.2): minima (+-1, 0, 0), constrained saddles (0, +-1, 0) with E = 2."""

    def real(self, f):
        return f

    def energy_gradient(self, X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        E = (x * x - 1) ** 2 + y * y + 2 * z * z
        return E, 0.5 * torch.stack([4 * x * (x * x - 1), 2 * y, 4 * z], dim=1)

    def inner(self, a, b):
        return (a * b).sum(dim=1)

    def preconditioner(self, X, mu):
        return (lambda f, i=slice(None): f), (lambda f, i=slice(None): self.inner(f, f))


@pytest.mark.parametrize("n", [9, 17])
def test_sphere_saddle_du_zhang(n):
    s = torch.linspace(0, 1, n, dtype=torch.float64)
    X = torch.stack([torch.cos(math.pi * s), 0.6 * torch.sin(math.pi * s), 0.8 * torch.sin(math.pi * s)], dim=1)
    Y, info = string_method(Sphere(), X, step=0.2, tol=1e-10)
    c = info["saddle"]
    assert float((Y[c] - torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)).abs().max()) < 1e-8, Y[c]
    assert abs(info["barrier"] - 2.0) < 1e-12, info["barrier"]
    assert float(Y[:, 2].abs().max()) < 1e-8                  # the path lies on the equator z = 0
    assert info["perpendicular"] < 1e-8, info


def _trap(dtype, g):
    grid = dipgpe.Grid(128, 16.0, dtype=dtype, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Contact(g))
    x = grid.x64[0]
    phi0 = (math.pi ** -0.25 * torch.exp(-x * x / 2)).to(dtype)
    chi = ((x + 0.3 * x ** 3) * torch.exp(-x * x / 2)).to(dtype)     # odd, not the exact saddle
    return grid, model, phi0, chi / torch.sqrt(dipgpe.norm(grid, chi))


def _string(model, minimum, chi, n, step=0.5):
    """String from the ground state to its negative (distinct minima among real fields) through chi."""
    space = FieldSpace(model)
    return string_method(space, path_through(space, [minimum, chi, -minimum], n), step=step)


@pytest.mark.parametrize("dtype", [C128, C64])
def test_linear_oscillator_saddle_is_first_excited_state(dtype):
    """g = 0: the saddle between +-phi0 is phi1 (index 1), barrier mu1 - mu0 = 1, and
    the Hessian on the tangent space at phi1 is H - mu1: eigenvalues -1, 1, 2."""
    grid, model, phi0, chi = _trap(dtype, 0.0)
    X, info = _string(model, phi0, chi, 9)
    c = info["saddle"]
    tol = 1e-12 if dtype == C128 else 1e-6
    assert abs(info["barrier"] - 1.0) < tol, info["barrier"]
    phi1 = (math.sqrt(2) * math.pi ** -0.25 * grid.x64[0] * torch.exp(-grid.x64[0] ** 2 / 2)).to(dtype)
    err = min(float(dipgpe.norm(grid, X[c] - phi1)), float(dipgpe.norm(grid, X[c] + phi1))) ** 0.5
    assert err < (1e-9 if dtype == C128 else 1e-4), err
    lam, _, _ = constrained_hessian(model, X[c], n_modes=3, tol=1e-6 if dtype == C128 else 1e-4)
    assert max(abs(a - b) for a, b in zip(lam.tolist(), [-1.0, 1.0, 2.0])) < (1e-10 if dtype == C128 else 1e-5), lam


@pytest.fixture(scope="module")
def dark_soliton_reference():
    """g = 5: the lowest odd stationary state (dark soliton), by minimization from
    an odd start (gradient steps keep the parity), and the ground state; complex128."""
    grid, model, phi0, chi = _trap(C128, 5.0)
    even, ie = dipgpe.ground_state(model, phi0)
    odd, io = dipgpe.ground_state(model, chi)
    flip = lambda f: torch.roll(torch.flip(f, [0]), 1)              # x -> -x on the grid  # noqa: E731
    assert float(dipgpe.norm(grid, odd + flip(odd))) < 1e-20         # stayed odd
    return float(io["energy"] - ie["energy"]), odd.cpu()


@pytest.mark.parametrize("dtype", [C128, C64])
def test_trapped_gas_saddle_is_dark_soliton(dtype, dark_soliton_reference):
    """g = 5: barrier against the odd-minimization reference to 1e-6 (relative), the
    saddle is the dark soliton, Morse index 1 there and 0 at the ground state, and
    doubling the images leaves the barrier unchanged."""
    barrier_ref, odd = dark_soliton_reference
    grid, model, phi0, chi = _trap(dtype, 5.0)
    even, _ = dipgpe.ground_state(model, phi0)
    barriers = []
    for n in (9, 17):
        X, info = _string(model, even, chi, n)
        assert info["perpendicular"] <= (1e-8 if dtype == C128 else 1e-5), info
        barriers.append(info["barrier"])
        assert abs(info["barrier"] - barrier_ref) < 1e-6 * barrier_ref, (n, info["barrier"], barrier_ref)
    assert abs(barriers[1] - barriers[0]) < (1e-12 if dtype == C128 else 1e-6) * barrier_ref, barriers
    c = info["saddle"]
    odd = odd.to(DEVICE, dtype)
    err = min(float(dipgpe.norm(grid, X[c] - odd)), float(dipgpe.norm(grid, X[c] + odd))) ** 0.5
    assert err < (1e-8 if dtype == C128 else 1e-4), err
    tol = 1e-6 if dtype == C128 else 1e-4
    lam, _, _ = constrained_hessian(model, X[c], n_modes=3, tol=tol)
    assert lam[0] < -0.1 and lam[1] > 0.1, lam                       # index 1
    lam0, _, _ = constrained_hessian(model, even, n_modes=2, tol=tol)
    assert lam0[0] > 0.1, lam0                                         # index 0


def test_string_converges_on_every_image():
    """g = 5: the normal gradient is below tolerance on every image for any step h
    (barrier independent of h), and the path converges as the images double
    (first order: upwind tangent)."""
    grid, model, phi0, chi = _trap(C128, 5.0)
    even, _ = dipgpe.ground_state(model, phi0)
    runs = [_string(model, even, chi, 17, step=h)[1] for h in (0.5, 0.25)]
    assert all(r["perpendicular"] <= 1e-8 for r in runs), runs
    assert abs(runs[0]["barrier"] - runs[1]["barrier"]) < 1e-12, runs
    paths = {n: _string(model, even, chi, n)[0] for n in (9, 17, 33)}
    dist = [max(float(dipgpe.norm(grid, paths[a][i] - paths[b][2 * i])) ** 0.5 for i in range(a))
            for a, b in ((9, 17), (17, 33))]
    assert dist[1] < 0.6 * dist[0] and dist[1] < 0.02, dist


def test_hessian_parity_sectors():
    """g = 0 at phi1: H - mu1 in the even sector gives -1 (phi0), +1 (phi2); in the
    odd sector (phi1 itself projected out) +2 (phi3), +4 (phi5)."""
    grid, model, _, _ = _trap(C128, 0.0)
    x = grid.x64[0]
    phi1 = (math.sqrt(2) * math.pi ** -0.25 * x * torch.exp(-x * x / 2)).to(C128)
    mirror = lambda f: torch.roll(torch.flip(f, [-1]), 1, dims=-1)          # noqa: E731
    for sign, exact in ((1.0, [-1.0, 1.0]), (-1.0, [2.0, 4.0])):
        lam, _, _ = constrained_hessian(model, phi1, n_modes=2, sector=lambda f, s=sign: 0.5 * (f + s * mirror(f)))
        assert max(abs(a - b) for a, b in zip(lam.tolist(), exact)) < 1e-9, (sign, lam)


@pytest.mark.parametrize("dtype", [C128, C64])
def test_newton_converges_to_first_excited_state(dtype):
    """g = 0: Newton on the sphere from phi1 + 0.1 (phi0 + phi3) converges to phi1 (E = 3/2)."""
    grid, model, phi0, _ = _trap(dtype, 0.0)
    x = grid.x64[0]
    phi1 = (math.sqrt(2) * math.pi ** -0.25 * x * torch.exp(-x * x / 2)).to(dtype)
    phi3 = (math.pi ** -0.25 / math.sqrt(48) * (8 * x ** 3 - 12 * x) * torch.exp(-x * x / 2)).to(dtype)
    start = phi1 + 0.1 * (phi0 + phi3)
    psi, info = refine_saddle(model, start / torch.sqrt(dipgpe.norm(grid, start)))
    assert info["converged"] and info["iterations"] <= 6, info
    err = min(float(dipgpe.norm(grid, psi - phi1)), float(dipgpe.norm(grid, psi + phi1))) ** 0.5
    assert err < (1e-9 if dtype == C128 else 1e-4), err
    assert abs(info["energy"] - 1.5) < (1e-12 if dtype == C128 else 1e-6), info["energy"]


@pytest.mark.parametrize("dtype", [C128, C64])
def test_newton_refines_coarse_string_saddle(dtype, dark_soliton_reference):
    """g = 5: a coarse string (climbing residual 1e-2) refined by Newton gives the
    dark soliton and the barrier to 1e-6, in a few steps, with Morse index 1."""
    barrier_ref, odd = dark_soliton_reference
    grid, model, phi0, chi = _trap(dtype, 5.0)
    even, ie = dipgpe.ground_state(model, phi0)
    space = FieldSpace(model)
    X, sinfo = string_method(space, path_through(space, [even, chi, -even], 9), tol=1e-3, residual_tol=1e-2)
    psi, info = refine_saddle(model, X[sinfo["saddle"]])
    assert info["converged"] and info["iterations"] <= 6, info
    assert all(b < a for a, b in zip(info["residuals"], info["residuals"][1:])), info["residuals"]
    barrier = info["energy"] - float(ie["energy"])
    assert abs(barrier - barrier_ref) < 1e-6 * barrier_ref, (barrier, barrier_ref)
    odd = odd.to(DEVICE, dtype)
    err = min(float(dipgpe.norm(grid, psi - odd)), float(dipgpe.norm(grid, psi + odd))) ** 0.5
    assert err < (1e-8 if dtype == C128 else 1e-4), err
    lam, _, _ = constrained_hessian(model, psi, n_modes=2, tol=1e-6 if dtype == C128 else 1e-4)
    assert lam[0] < -0.1 and lam[1] > 0.1, lam


def test_newton_from_a_rough_guess():
    """g = 5, complex128: from the odd guess chi (residual 0.6) to the dark soliton."""
    grid, model, _, chi = _trap(C128, 5.0)
    psi, info = refine_saddle(model, chi)
    assert info["converged"] and info["iterations"] <= 10, info


def test_callback_stops_the_string():
    s = torch.linspace(0, 1, 9, dtype=torch.float64)
    X = torch.stack([torch.cos(math.pi * s), 0.6 * torch.sin(math.pi * s), 0.8 * torch.sin(math.pi * s)], dim=1)
    _, info = string_method(Sphere(), X, step=0.2, callback=lambda it, X, info: it >= 5, callback_every=1)
    assert info["stopped"] and not info["converged"] and info["iterations"] == 5, info


def test_rotation_is_refused():
    grid = dipgpe.Grid((16, 16), 8.0, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Potential(dipgpe.harmonic(grid)), dipgpe.Rotation(0.5))
    with pytest.raises(NotImplementedError):
        FieldSpace(model)


def test_break_symmetry_double_well():
    """Attractive gas in a double well at fixed mu (grand canonical), a model of C3-2 (e):
    the symmetric state has index 2 (norm mode and odd mode unstable). Newton from it
    returns to it; break_symmetry along the odd mode reaches the self-trapped state,
    which the fixed-N minimization finds independently at the same mu, with index 1."""
    grid = dipgpe.Grid(128, 16.0, dtype=C128, device=DEVICE)
    x = grid.x64[0]
    V = 0.5 * x * x + 5.0 * torch.exp(-x * x / 0.72)
    model = dipgpe.GPE(grid, dipgpe.Potential(V), dipgpe.Contact(-1.0))
    mirror = lambda f: torch.roll(torch.flip(f, [-1]), 1, dims=-1)          # noqa: E731
    start = torch.exp(-(x + 1.5) ** 2).to(C128)
    selftrapped, _ = dipgpe.ground_state(model, start / torch.sqrt(dipgpe.norm(grid, start)))
    mu = float(dipgpe.chemical_potential(model, selftrapped))
    sym, _ = refine_saddle(model, selftrapped + mirror(selftrapped), mu=mu, sector=lambda f: 0.5 * (f + mirror(f)))
    lam, modes, _ = constrained_hessian(model, sym, n_modes=3, mu=mu)
    assert lam[0] < -0.5 and lam[1] < -0.5 and lam[2] > 0.5, lam
    odd = max(modes, key=lambda m: float((m - mirror(m)).abs().norm()))
    out, info = break_symmetry(model, sym, odd, [0.1, 0.2, 0.35, 0.5, 0.65, 0.8], mu=mu)
    assert info["converged"] and info["scan"][0]["beta"] < 0, info
    err = min(float(dipgpe.norm(grid, out - s * f)) for s in (1, -1) for f in (selftrapped, mirror(selftrapped)))
    assert err ** 0.5 < 1e-8, err
    lam, _, _ = constrained_hessian(model, out, n_modes=2, mu=mu)
    assert lam[0] < -0.5 and lam[1] > 0.1, lam


def test_descend_connects_both_sides():
    """Connectivity at fixed mu: in the attractive double well the self-trapped state has
    index 1, its unstable mode changing N (dN/dmu < 0). Descending from it on either side
    of that mode reaches psi = 0 (Omega = 0, a minimum since mu lies below the lowest
    single-particle level) on one side, and a state of growing N and falling Omega on
    the other (Omega is unbounded below for an attractive 1D gas at fixed mu)."""
    grid = dipgpe.Grid(128, 16.0, dtype=C128, device=DEVICE)
    x = grid.x64[0]
    V = 0.5 * x * x + 5.0 * torch.exp(-x * x / 0.72)
    model = dipgpe.GPE(grid, dipgpe.Potential(V), dipgpe.Contact(-1.0))
    start = torch.exp(-(x + 1.5) ** 2).to(C128)
    psi, ie = dipgpe.ground_state(model, start / torch.sqrt(dipgpe.norm(grid, start)))
    mu = float(dipgpe.chemical_potential(model, psi))
    _, il = dipgpe.ground_state(dipgpe.GPE(grid, dipgpe.Potential(V)), start / torch.sqrt(dipgpe.norm(grid, start)))
    assert mu < float(il["energy"]), (mu, il["energy"])            # psi = 0 is a minimum of Omega
    omega = float(ie["energy"]) - mu * 1.0
    lam, modes, _ = constrained_hessian(model, psi, n_modes=2, mu=mu)
    assert lam[0] < -0.5 < 0.1 < lam[1], lam
    v = modes[0].real.to(C128)
    v = v / torch.sqrt(dipgpe.norm(grid, v))
    if float((v.conj() * psi).sum().real) < 0:
        v = -v                                                     # +v adds atoms
    down, info = descend(model, psi - 0.1 * v, mu=mu, callback=lambda it, f, i: i["N"] < 1e-8, callback_every=1)
    assert info["stopped"] and info["omega"] < 1e-8, info
    up, info = descend(model, psi + 0.1 * v, mu=mu, callback=lambda it, f, i: i["N"] > 1.5, callback_every=1)
    assert info["stopped"] and info["omega"] < omega - 0.05, (info, omega)


def test_break_symmetry_steps_from_a_minimum_to_a_saddle():
    """break_symmetry as one step of the C3-2 (e) staircase (any direction, either sign
    of the multiplier): from the minimum psi = 0 of the attractive double well at fixed
    mu, along the self-trapped state, the multiplier is positive (uphill) until it
    changes sign at that state (index 1), which the step reaches."""
    grid = dipgpe.Grid(128, 16.0, dtype=C128, device=DEVICE)
    x = grid.x64[0]
    V = 0.5 * x * x + 5.0 * torch.exp(-x * x / 0.72)
    model = dipgpe.GPE(grid, dipgpe.Potential(V), dipgpe.Contact(-1.0))
    start = torch.exp(-(x + 1.5) ** 2).to(C128)
    selftrapped, _ = dipgpe.ground_state(model, start / torch.sqrt(dipgpe.norm(grid, start)))
    mu = float(dipgpe.chemical_potential(model, selftrapped))
    out, info = break_symmetry(model, torch.zeros_like(selftrapped), selftrapped, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                               mu=mu)
    assert info["converged"] and info["scan"][0]["beta"] > 0, info
    err = min(float(dipgpe.norm(grid, out - s * selftrapped)) for s in (1, -1)) ** 0.5
    assert err < 1e-8, err
