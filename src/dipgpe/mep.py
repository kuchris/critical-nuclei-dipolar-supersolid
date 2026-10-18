"""Minimum energy paths and saddle points at fixed norm: the string method.

A string of images psi_0 ... psi_n on the sphere ||psi||^2 = N joins two minima.
Each iteration (string method of E, Ren & Vanden-Eijnden, Phys. Rev. B 66,
052301 (2002) and J. Chem. Phys. 126, 164103 (2007); the constraint by its
Lagrange multiplier as in Du & Zhang, Commun. Math. Sci. 7, 1039 (2009)):

1. every interior image moves along the part of its preconditioned gradient
   g = P (H psi - lambda psi) (tangent to the sphere, the direction of
   minimize.py) normal to the path, psi -> psi - h (g - <g, t> t / <t, t>),
   then the exact norm is restored (Du & Zhang's general string dynamics,
   Eq. (3.5));
2. the images are redistributed at equal arc length (L2 distance) along the
   natural cubic spline through them.

The tangent t is the upwind one of Henkelman & Jonsson (J. Chem. Phys. 113,
9978 (2000)): the difference to the higher-energy neighbour, an
energy-weighted mixture at extrema. A centred (spline) tangent makes the
explicit step an unstable centred scheme for the advection-like coupling
between the tangential gradient and the tilt of the tangent: zig-zag strings
that did not converge (ISSUES C3-2). Moving with the full g instead (the
simplified method) is stable but leaves a normal gradient of order h in the
converged string (ISSUES C3-3). With the upwind tangent the converged string
has g parallel to t on every image.

The preconditioner is the combined one of minimize.py, P = P_V^1/2 P_k P_V^1/2
with P_k = (|mu| + k^2/2)^-1 and P_V = |mu| / (|mu| + U - min U) (U the local
potential of the image), normalized so that P ~ 1/|mu| where the potential is
low: the fixed step h is then stable for h of order 1 also where a trap makes U
large. The string converges to a minimum energy path in the metric P^-1 (P g
parallel to the path); its saddle point and the barrier do not depend on that
metric. Once the string is close to converged, its highest interior image
climbs (Ren & Vanden-Eijnden, J. Chem. Phys. 138, 134105 (2013)): its full
gradient is reflected along the tangent t in the same metric, g -> g - 2 <r, t>
/ <t, P^-1 t> t with r = H psi - mu psi, so it converges to the saddle point
itself (r = 0) while the other images are redistributed on each side of it.

Images are real fields: static paths without currents, appropriate for an
energy without rotation, whose saddle points between real states are real.
Restricting to real fields also removes the gauge orbit, along which psi and
-psi are joined at no energy cost.

The climbing image converges only at the rate of the softest mode at the
saddle (thousands of iterations in the 3D tube, ISSUES C3-5);
:func:`refine_saddle` finishes it by Newton's method (MINRES on the
indefinite Hessian), in a few steps; with linear constraints (Lagrange
multipliers) it also gives constrained stationary states, which
:func:`break_symmetry` uses to reach a symmetry-broken stationary state from a
symmetric one.

The algorithm only needs the operations of a *space* (``FieldSpace`` for a GPE
model; any object with the same methods, e.g. vectors in R^n for tests):
    real(f)                  the field restricted to the allowed (real) subspace,
    energy_gradient(X)    -> (E (n,), h), h proportional to the gradient
                             (for fields h = dE/dpsi*),
    inner(a, b)           -> (n,) per-image inner products,
    preconditioner(X, mu) -> (apply(f, i), metric(f, i) = <f, P^-1 f>) for the
                             images i (an index or slice) of X.
"""

from __future__ import annotations

import bisect
import math

import torch

from .bdg import Bogoliubov, _add_deflation
from .diagnostics import density
from .minimize import _expand, _inner, energy_and_gradient


class FieldSpace:
    """Real fields of a GPE model (images on the leading axis)."""

    def __init__(self, model, t=0.0):
        for term in model.terms:
            if not hasattr(term, "local"):
                raise NotImplementedError(f"the string method needs local terms only, not {type(term).__name__}")
        self.model, self.grid, self.t = model, model.grid, t

    def real(self, f):
        return f.real.to(self.grid.dtype)

    def energy_gradient(self, X):
        E, h = energy_and_gradient(self.model, X, self.t)
        return E, self.real(h)

    def inner(self, a, b):
        return _inner(self.grid, a, b)

    def preconditioner(self, X, mu):
        """(apply, metric) of P = P_V^1/2 P_k P_V^1/2 at the images X (see the module docstring)."""
        g = self.grid
        m = _expand(mu.abs(), g)
        U = self.model.local_potential(density(X), self.t)
        if torch.is_tensor(U):
            U = U - U.amin(dim=g.dims, keepdim=True)
            pv = torch.sqrt(m / (m + U)).to(g.real_dtype)
        else:
            pv = torch.ones_like(m)
        inv = 1.0 / (m + 0.5 * g.k2)

        def apply(f, i=slice(None)):
            return self.real(pv[i] * g.ifft(g.fft(pv[i] * f) * inv[i]))

        def metric(f, i=slice(None)):
            w = f / pv[i]
            kin = 0.5 * (g.k2 * density(g.fft(w))).sum(dim=g.dims, dtype=torch.float64) * (g.dV / g.size)
            return mu[i].abs() * self.inner(w, w) + kin

        return apply, metric


def _residual_tol(grid):
    """The default residual tolerance of :func:`dipgpe.ground_state`: 1e-9 (complex128)
    or 1e-5 (complex64), raised to 4x the FFT round-off floor k^2_max eps / 2."""
    floor = 0.5 * sum((math.pi / dx) ** 2 for dx in grid.dx) * torch.finfo(grid.real_dtype).eps
    return max(1e-5 if grid.dtype == torch.complex64 else 1e-9, 4 * floor)


def _interval(s, t):
    """Index j of the interval [s_j, s_j+1] containing t (clamped to the first/last)."""
    return min(max(bisect.bisect_right(s.tolist(), t) - 1, 0), len(s) - 2)


def _spline_weights(s, targets):
    """Matrix W (float64) with W @ y the natural cubic spline through (s_j, y_j)
    at ``targets``; linear in y, so it applies to fields."""
    n = len(s)
    if n < 3:
        raise ValueError("a string needs at least 3 images")
    dx = s[1:] - s[:-1]
    # second derivatives m = S y: natural ends, continuity of the first derivative inside
    T = torch.zeros(n, n, dtype=torch.float64)
    R = torch.zeros(n, n, dtype=torch.float64)
    T[0, 0] = T[-1, -1] = 1.0
    for i in range(1, n - 1):
        T[i, i - 1], T[i, i], T[i, i + 1] = dx[i - 1] / 6, (dx[i - 1] + dx[i]) / 3, dx[i] / 6
        R[i, i - 1], R[i, i], R[i, i + 1] = 1 / dx[i - 1], -1 / dx[i - 1] - 1 / dx[i], 1 / dx[i]
    S = torch.linalg.solve(T, R)
    eye = torch.eye(n, dtype=torch.float64)
    W = torch.zeros(len(targets), n, dtype=torch.float64)
    for r, t in enumerate(targets.tolist()):
        j = _interval(s, t)
        hj = float(dx[j])
        a = (float(s[j + 1]) - t) / hj
        b = 1.0 - a
        W[r] = a * eye[j] + b * eye[j + 1] + ((a ** 3 - a) * S[j] + (b ** 3 - b) * S[j + 1]) * hj * hj / 6
    return W


def _apply(W, X):
    return torch.einsum("ij,j...->i...", W.to(device=X.device, dtype=X.real.dtype).to(X.dtype), X)


def _arc(space, X):
    d = X[1:] - X[:-1]
    return torch.cat([torch.zeros(1, dtype=torch.float64),
                      torch.cumsum(torch.sqrt(space.inner(d, d)).cpu().to(torch.float64), 0)])


def _targets(s, climb):
    if climb is None:
        return torch.linspace(0, float(s[-1]), len(s), dtype=torch.float64)
    c = climb
    left = torch.linspace(0, float(s[c]), c + 1, dtype=torch.float64)
    right = torch.linspace(float(s[c]), float(s[-1]), len(s) - c, dtype=torch.float64)
    return torch.cat([left, right[1:]])


def _scale(space, X, N):
    return torch.sqrt(N / space.inner(X, X)).to(X.real.dtype).reshape(-1, *([1] * (X.dim() - 1)))


def _expand_images(c, X):
    return c.to(X.real.dtype).reshape(-1, *([1] * (X.dim() - 1)))


def _direction(space, P, X, h):
    """g = P (h - lambda X), tangent to the sphere (<X, g> = 0)."""
    Ph, PX = P(h), P(X)
    return Ph - _expand_images(space.inner(X, Ph) / space.inner(X, PX), X) * PX


def _tangents(space, X, E):
    """Upwind tangents (Henkelman & Jonsson 2000) at the interior images, made
    tangent to the sphere; zero at the ends."""
    E = E.to(torch.float64).cpu()
    n = X.shape[0]
    a = torch.zeros(n, dtype=torch.float64)                # weight of X[i+1] - X[i]
    b = torch.zeros(n, dtype=torch.float64)                # weight of X[i] - X[i-1]
    for i in range(1, n - 1):
        up, down = float(E[i + 1] - E[i]), float(E[i - 1] - E[i])
        if up > 0 > down:
            a[i] = 1.0
        elif up < 0 < down:
            b[i] = 1.0
        else:
            big, small = max(abs(up), abs(down)), min(abs(up), abs(down))
            a[i], b[i] = (big, small) if E[i + 1] > E[i - 1] else (small, big)
    fwd = torch.zeros_like(X)
    bwd = torch.zeros_like(X)
    fwd[:-1] = X[1:] - X[:-1]
    bwd[1:] = X[1:] - X[:-1]
    t = _expand_images(a.to(X.device), X) * fwd + _expand_images(b.to(X.device), X) * bwd
    return t - _expand_images(space.inner(X, t) / space.inner(X, X), X) * X


def path_through(space, states, n_images):
    """Initial string: piecewise-linear interpolation through ``states`` at equal
    arc length, each image normalized to the norm of the first state."""
    X = torch.stack([space.real(s) for s in states])
    s = _arc(space, X)
    t = torch.linspace(0, float(s[-1]), n_images, dtype=torch.float64)
    W = torch.zeros(n_images, len(states), dtype=torch.float64)
    for r, v in enumerate(t.tolist()):
        j = _interval(s, v)
        a = (float(s[j + 1]) - v) / float(s[j + 1] - s[j])
        W[r, j], W[r, j + 1] = a, 1.0 - a
    Y = _apply(W, X)
    return Y * _scale(space, Y, space.inner(X[:1], X[:1]))


def string_method(space, images, *, step=0.5, tol=None, residual_tol=None, climb=True, climb_tol=1e-2,
                  fixed_ends=True, sector=None, max_iter=100_000, callback=None, callback_every=100, strict=True):
    """Minimum energy path through ``images`` (leading axis; the first and last
    should be minima) and, with ``climb``, the saddle point on it.

    Every image keeps the norm of the first. ``step`` is h (see the module
    docstring). The string has converged when the normal part of the
    preconditioned gradient, ||g - <g, t> t / <t, t>|| / sqrt(N), is below ``tol``
    on every interior image other than the climbing one (default 1e-8, or 1e-5
    for complex64 fields), and the climbing image when its residual
    ||H psi - mu psi|| / sqrt(N) is below ``residual_tol`` (default: the
    :func:`dipgpe.ground_state` tolerance for fields, 1e-9 otherwise).
    Climbing starts when the normal part falls below ``climb_tol``; the
    climbing image is the highest interior image at that moment. ``fixed_ends``
    keeps the end images; otherwise they relax to their minima. ``sector`` (a
    linear projection commuting with the energy, e.g. onto fields even in z) is
    applied to the images at the start and after every update.
    ``callback(iteration, images, info)`` is called every ``callback_every``
    iterations; if it returns True the iteration stops there (info["stopped"]
    True, not converged), e.g. at a time budget.

    Returns (images, info): info = {"energies" (float64, per image), "saddle"
    (index of the climbing image or None), "barrier" (E_saddle - E_0, or the
    highest energy minus E_0 without climbing), "residual" (of the climbing
    image), "perpendicular" (the largest normal part), "iterations",
    "converged", "stopped"}. If not converged within ``max_iter``, raises
    RuntimeError, or returns with converged False when ``strict`` is False.
    """
    X = space.real(images if sector is None else sector(images))
    n = X.shape[0]
    N = space.inner(X[:1], X[:1])
    X = X * _scale(space, X, N)
    sqrtN = math.sqrt(float(N))
    grid = getattr(space, "grid", None)
    single = grid is not None and grid.dtype == torch.complex64
    if tol is None:
        tol = 1e-5 if single else 1e-8
    if residual_tol is None:
        residual_tol = _residual_tol(grid) if grid is not None else 1e-9
    c = None
    res_c = None
    stop = False
    for it in range(max_iter + 1):
        E, h = space.energy_gradient(X)
        nx = space.inner(X, X)
        mu = space.inner(X, h) / nx
        r = h - _expand_images(mu, X) * X
        P, metric = space.preconditioner(X, mu)
        g = _direction(space, P, X, h)
        t = _tangents(space, X, E)
        tt = space.inner(t, t)
        tt[0] = tt[-1] = 1.0                           # ends: t = 0
        normal = g - _expand_images(space.inner(g, t) / tt, X) * t
        others = [i for i in range(1, n - 1) if i != c]
        perp = float(torch.sqrt(space.inner(normal[others], normal[others])).max()) / sqrtN if others else 0.0
        if c is not None:
            res_c = float(torch.sqrt(space.inner(r[c:c + 1], r[c:c + 1]))[0]) / sqrtN
        done = perp <= tol and (not climb or (c is not None and res_c <= residual_tol))
        if callback is not None and it % callback_every == 0 and not done:
            stop = bool(callback(it, X, {"energies": E, "saddle": c, "residual": res_c, "perpendicular": perp}))
        if done or stop or it == max_iter:
            break
        if climb and c is None and perp <= climb_tol:
            c = int(torch.argmax(E[1:-1])) + 1
        d = normal
        d[0], d[-1] = (0, 0) if fixed_ends else (g[0], g[-1])
        if c is not None:                              # reflect along the tangent (module docstring)
            i = slice(c, c + 1)
            coef = space.inner(r[i], t[i]) / metric(t[i], i)
            d[i] = g[i] - 2 * _expand_images(coef, t[i]) * t[i]
        Y = X - step * d
        if sector is not None:
            Y = space.real(sector(Y))
        Y = Y * _scale(space, Y, N)
        s = _arc(space, Y)
        W = _spline_weights(s, _targets(s, c))
        for j in (0, n - 1, c):
            if j is not None:                          # ends and climbing image exactly kept
                W[j] = 0
                W[j, j] = 1.0
        Y = _apply(W, Y)
        X = space.real(Y * _scale(space, Y, N))
    if not done and not stop and strict:
        raise RuntimeError(f"string_method: normal gradient {perp:.3g}, saddle residual {res_c} "
                           f"after {max_iter} iterations")
    E64 = E.to(torch.float64).cpu()
    barrier = float(E64[c] - E64[0]) if c is not None else float(E64.max() - E64[0])
    return X, {"energies": E64, "saddle": c, "barrier": barrier, "residual": res_c, "perpendicular": perp,
               "iterations": it, "converged": done, "stopped": stop}


def _minres(apply_A, apply_Minv, b, inner, tol, max_iter):
    """Preconditioned MINRES (Paige & Saunders; as in scipy.sparse.linalg.minres) for
    A x = b with A symmetric (indefinite allowed) and M^-1 symmetric positive
    definite, x0 = 0. Stops when the residual in the M^-1 norm has fallen by
    ``tol``. Returns (x, iterations, relative residual)."""
    x = torch.zeros_like(b)
    r1 = b
    y = apply_Minv(r1)
    beta1 = inner(r1, y)
    if beta1 <= 0:
        return x, 0, 0.0
    beta1 = math.sqrt(beta1)
    beta, oldb, dbar, epsln, phibar, cs, sn = beta1, 0.0, 0.0, 0.0, beta1, -1.0, 0.0
    w = w2 = torch.zeros_like(b)
    r2 = r1
    for itn in range(1, max_iter + 1):
        v = y / beta
        y = apply_A(v)
        if itn >= 2:
            y = y - (beta / oldb) * r1
        alfa = inner(v, y)
        y = y - (alfa / beta) * r2
        r1, r2 = r2, y
        y = apply_Minv(r2)
        oldb, beta = beta, inner(r2, y)
        if beta < 0:
            raise RuntimeError("minres: the preconditioner is not positive definite")
        beta = math.sqrt(beta)
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        gamma = max(math.hypot(gbar, beta), 1e-300)
        cs, sn = gbar / gamma, beta / gamma
        phi = cs * phibar
        phibar = sn * phibar
        w1, w2 = w2, w
        w = (v - oldeps * w1 - delta * w2) / gamma
        x = x + phi * w
        if phibar <= tol * beta1 or beta == 0:
            break
    return x, itn, phibar / beta1


def refine_saddle(model, psi, *, mu=None, tol=None, linear_tol=1e-2, max_newton=30, max_linear=5000, deflate=None,
                  t=0.0, sector=None, strict=True, callback=None):
    """Stationary state near ``psi`` (real; its norm is kept) by Newton's method on
    the sphere, for any Morse index: e.g. the saddle point from
    :func:`string_method`, whose climbing image converges only at the rate of the
    softest mode.

    With ``mu`` the chemical potential is fixed instead of the norm (grand
    canonical): Newton solves H psi = mu psi, a stationary point of E - mu N on all
    real fields, with no projection off psi and no renormalization. This is the
    saddle of a nucleus exchanging atoms with an unbounded background at mu, i.e.
    the L -> infinity limit of the fixed-N tube without its depletion term
    (ISSUES C3-13). info["N"] is the norm of the result.

    ``deflate`` holds linear constraints: the components <c, psi> of these fields
    keep their start values (each step is orthogonal to them; at fixed N the
    renormalization changes them at second order). The result is stationary up to
    forces along them, H psi - mu psi = sum_c eta_c c, with the Lagrange
    multipliers eta in info["multipliers"]; the residual excludes those forces and
    info["full_residual"] includes them. For a symmetry direction (d psi/dz of a
    nucleus in a periodic box, taken from a reference state: a phase condition)
    eta vanishes at a true stationary state, since <d psi/dz, H psi - mu psi> = 0
    for every psi; for any other field (e.g. a fixed antisymmetric amplitude,
    :func:`break_symmetry`) the result is a constrained stationary state and eta
    the slope of the energy along that field.

    Each step solves P M P delta = -(H psi - mu psi) on the tangent space, with M
    the Hessian of E - mu N for real perturbations (:class:`dipgpe.bdg.Bogoliubov`,
    the operator M = L + 2X at the current psi) and P the projection off psi
    (at fixed N) and off the constraint fields. M is
    indefinite at a saddle, so the step is solved by MINRES with the kinetic
    preconditioner (|k|^2/2 + max(|mu|, 1))^-1, to relative accuracy ``linear_tol``
    (inexact Newton: the residual falls by about that factor per step once the
    quadratic term is small). psi + delta is renormalized; if the residual did not
    fall the step is halved (down to 1/64). If a step fails, or lowers the residual
    by less than half, the linear tolerance is tightened 100-fold (down to 1e-8; in
    the 3D tube a 1e-2 solve left floors of 1e-8 to 1e-6, ISSUES C3-9); if that does
    not help either, the iteration stops at the best state (info["stalled"]).
    Stops when the residual
    ||H psi - mu psi|| / ||psi|| is below ``tol`` (default as
    :func:`dipgpe.ground_state`). ``sector`` (a linear projection commuting with
    the energy, e.g. onto fields even in z) is applied to the start and to every
    step, so the iteration stays in that symmetry sector: in a periodic tube, a
    near-zero odd mode otherwise lets parity-breaking parts of ~1e-3 survive
    (ISSUES C3-10). ``callback(step, psi, residual)`` is called after every Newton
    step. Check the Morse index with :func:`dipgpe.bdg.constrained_hessian`.

    Returns (psi, info): info = {"iterations", "residual", "full_residual",
    "multipliers", "energy", "mu", "N", "residuals" (per Newton step),
    "linear_iterations", "linear_tolerances", "converged", "stalled"}. If not
    converged within ``max_newton`` steps, raises RuntimeError, or returns the
    last state with converged False when ``strict`` is False.
    """
    space = FieldSpace(model, t)
    grid = model.grid
    if tol is None:
        tol = _residual_tol(grid)
    psi = space.real(psi)
    N = float(space.inner(psi[None], psi[None])[0])
    ip = lambda a, b: float(_inner(grid, a, b))          # noqa: E731
    fixed_mu = mu
    if sector is not None:
        psi = space.real(sector(psi))
        if fixed_mu is None:
            psi = psi * math.sqrt(N / ip(psi, psi))
    cons = [space.real(c.to(psi.device)) for c in (deflate or [])]

    def state(f):
        """Energy, multipliers (mu at fixed N, eta of the constraints), residual without and with the constraint forces."""
        E, h = space.energy_gradient(f[None])
        h = h[0]
        n = N if fixed_mu is None else ip(f, f)
        basis = ([f] if fixed_mu is None else []) + cons
        g = h if fixed_mu is None else h - fixed_mu * f
        coef = []
        if basis:                                       # least squares: g - sum coef basis orthogonal to the basis
            G = torch.tensor([[ip(a, b) for b in basis] for a in basis], dtype=torch.float64)
            coef = torch.linalg.solve(G, torch.tensor([ip(a, g) for a in basis], dtype=torch.float64)).tolist()
        r = g
        for c_, b in zip(coef, basis):
            r = r - c_ * b
        m = coef[0] if fixed_mu is None else fixed_mu
        multipliers = coef[1:] if fixed_mu is None else coef
        r_full = h - m * f
        return float(E[0]), m, r, math.sqrt(ip(r, r) / n), multipliers, math.sqrt(ip(r_full, r_full) / n)

    def renormalized(f):
        return f * math.sqrt(N / ip(f, f)) if fixed_mu is None else f

    E, mu, r, res, mult, res_full = state(psi)
    residuals, linear, etas = [res], [], []
    stalled = False
    eta = linear_tol
    for k in range(max_newton + 1):
        if res <= tol or k == max_newton:
            break
        op = Bogoliubov(model, psi, t=t)
        op.mu = mu
        _add_deflation(op, ([psi] if fixed_mu is None else []) + cons)
        if sector is not None:
            project = op.project
            op.project = lambda f, project=project: project(sector(f))
        shift = max(abs(op.mu), 1.0)
        A = lambda f: op.project(op.M(op.project(f)))                                  # noqa: E731
        Minv = lambda f: op.project(op.precondition(op.project(f), shift))             # noqa: E731
        delta, its, _ = _minres(A, Minv, -op.project(r), ip, eta, max_linear)
        delta = space.real(delta)
        linear.append(its)
        etas.append(eta)
        lam = 1.0
        while True:
            trial = renormalized(psi + lam * delta)
            st = state(trial)
            res_t = st[3]
            if res_t < res or lam <= 1 / 64:
                break
            lam /= 2
        slow = res_t > 0.5 * res
        if res_t < res:
            psi, (E, mu, r, res, mult, res_full) = trial, st
            residuals.append(res)
            if callback is not None:
                callback(k + 1, psi, res)
        if slow and res > tol:
            if eta <= 1e-8:
                stalled = True
                break
            eta = max(eta * 1e-2, 1e-8)
    converged = res <= tol
    if not converged and strict:
        raise RuntimeError(f"refine_saddle: residual {res:.3g} > tol {tol:g} after {len(residuals) - 1} Newton "
                           f"steps{' (stalled)' if stalled else ''}")
    return psi, {"iterations": len(residuals) - 1, "residual": res, "full_residual": res_full,
                 "multipliers": mult, "energy": E, "mu": mu, "N": ip(psi, psi),
                 "residuals": residuals, "linear_iterations": linear, "linear_tolerances": etas,
                 "converged": converged, "stalled": stalled}


def break_symmetry(model, psi, mode, amplitudes, *, mu=None, deflate=None, tol=None, max_newton=30,
                   secant_steps=6, t=0.0, callback=None):
    """Symmetry-broken stationary state next to a symmetric one (C3-2 (e); reviews
    in docs/reports/C3e_codex_answer.md and C3e_answer2.md).

    ``psi`` is a stationary state and ``mode`` an unstable direction that breaks its
    symmetry (e.g. the odd sliding mode of a symmetric nucleus of index 2). Newton
    from psi + eps * mode returns to psi, so instead the amplitude
    a = <mode, psi'> - <mode, psi> (mode normalized and made orthogonal to
    ``deflate``) is held by one more constraint of :func:`refine_saddle`, which
    relaxes everything else (``deflate``, e.g. a translation phase condition, is
    kept). The multiplier beta(a) of that constraint is the slope of the energy
    along the constrained branch: beta ~ lambda a near psi (lambda < 0 the
    eigenvalue of the mode), and it changes sign at a symmetry-broken stationary
    state. The same search steps from any stationary state to the next one along
    a given direction, whatever the sign of beta at the start (from a minimum to
    the next saddle and on: the staircase of tube_saddle.py). The ``amplitudes``
    (increasing) are continued until beta changes sign
    (a solve that does not converge is retried halfway from the last converged
    amplitude, up to four times); regula falsi (Illinois) on beta narrows the
    bracket (at most ``secant_steps``
    solves), and Newton without the amplitude constraint polishes the best state.

    Returns (psi, info): info = {"scan": [{"a", "beta", "residual", "energy", "N",
    "converged"}] (every constrained solve, in order), "bracket" (amplitudes or
    None), "a" (amplitude of the result), "polish" (refine_saddle info or None),
    "converged"}; converged requires a sign change of beta, a converged polish
    and a result that kept at least half of its amplitude (did not fall back to
    psi). Check its Morse index with :func:`dipgpe.bdg.constrained_hessian`
    (no symmetry sector, translation deflated).
    """
    space = FieldSpace(model, t)
    grid = model.grid
    ip = lambda a, b: float(_inner(grid, a, b))          # noqa: E731
    psi = space.real(psi)
    cons = [space.real(c.to(psi.device)) for c in (deflate or [])]
    v = space.real(mode.to(psi.device))
    for c in cons:
        v = v - (ip(c, v) / ip(c, c)) * c
    v = v / math.sqrt(ip(v, v))
    a0 = ip(v, psi)
    scan = []

    def solve(a, start):
        start = start + (a - (ip(v, start) - a0)) * v
        out, info = refine_saddle(model, start, mu=mu, tol=tol, max_newton=max_newton, deflate=cons + [v], t=t,
                                  strict=False)
        row = {"a": ip(v, out) - a0, "beta": info["multipliers"][-1], "residual": info["residual"],
               "energy": info["energy"], "N": info["N"], "converged": info["converged"]}
        scan.append(row)
        if callback is not None:
            callback(row)
        return out, row

    prev, bracket, start = None, None, psi
    pending, retries = list(amplitudes), 0
    while pending:
        a = pending.pop(0)
        out, row = solve(a, start)
        if not row["converged"]:                        # retry halfway from the last converged amplitude
            if prev is None or retries == 4:            # (at 90.27 a0 one solve past the root failed)
                break
            retries += 1
            pending[:0] = [0.5 * (prev[1]["a"] + a), a]
            continue
        if prev is not None and (row["beta"] > 0) != (prev[1]["beta"] > 0):
            bracket = (prev, (out, row))
            break
        prev, start = (out, row), out
    info = {"scan": scan, "bracket": None, "a": None, "polish": None, "converged": False}
    if bracket is None:
        return (psi if prev is None else prev[0]), info
    (lo, r_lo), (hi, r_hi) = bracket
    a_lo, f_lo, a_hi, f_hi = r_lo["a"], r_lo["beta"], r_hi["a"], r_hi["beta"]
    info["bracket"] = (a_lo, a_hi)
    best = min(bracket, key=lambda s: abs(s[1]["beta"]))
    scale = max(abs(r["beta"]) for r in scan)
    side = 0
    for _ in range(secant_steps):
        if abs(best[1]["beta"]) <= 1e-3 * scale:
            break
        a = (a_lo * f_hi - a_hi * f_lo) / (f_hi - f_lo)
        out, row = solve(a, lo if abs(a - a_lo) < abs(a - a_hi) else hi)
        if not row["converged"]:
            break
        if abs(row["beta"]) < abs(best[1]["beta"]):
            best = (out, row)
        if (row["beta"] > 0) == (f_lo > 0):
            lo, a_lo, f_lo = out, row["a"], row["beta"]
            if side == -1:
                f_hi /= 2
            side = -1
        else:
            hi, a_hi, f_hi = out, row["a"], row["beta"]
            if side == 1:
                f_lo /= 2
            side = 1
        info["bracket"] = (a_lo, a_hi)
    out, polish = refine_saddle(model, best[0], mu=mu, tol=tol, max_newton=max_newton, deflate=cons, t=t,
                                strict=False)
    a_out = ip(v, out) - a0
    info.update(a=a_out, polish=polish,
                converged=bool(polish["converged"] and abs(a_out) >= 0.5 * abs(best[1]["a"])))
    return out, info


def descend(model, psi, *, mu, max_iter=20_000, tau=0.5, tol=None, deflate=None, sector=None, t=0.0,
            callback=None, callback_every=50):
    """Downhill relaxation of Omega = E - mu N at fixed chemical potential (no norm
    constraint), for the connectivity check of a saddle (C3-2 (e); review
    docs/reports/C3e_codex_answer.md): started on both sides of its unstable mode,
    one run should reach the metastable state and the other the growing product.
    At fixed N use :func:`dipgpe.ground_state`.

    Nesterov-accelerated gradient descent with gradient restart (O'Donoghue &
    Candes, Found. Comput. Math. 15, 715 (2015)) on real fields: the gradient
    H psi - mu psi is preconditioned by the combined preconditioner of
    :class:`FieldSpace` (kinetic and local potential, refreshed every 20
    iterations; the kinetic one alone needed steps below 0.1 in a trap), kept in
    ``sector`` and orthogonal to ``deflate`` (fixed fields whose components keep
    their start values, e.g. a translation phase condition); a step changes the
    field by at most 10% of its norm (then the momentum restarts), and if Omega
    rises the iteration returns to the last accepted point with half the step (in
    the 3D tube a fixed step 0.5 diverged; the step recovers after 100 accepted
    iterations), so the accepted Omega never increases. It is a descent path,
    not the physical dynamics. Stops when the residual ||H psi - mu psi|| / ||psi||
    is below ``tol`` (default as :func:`refine_saddle`; a minimum or another
    stationary state), when ``callback(iteration, psi, info)`` returns True (e.g.
    once the product is recognized), when Omega stops being finite, or after
    ``max_iter`` iterations.

    Returns (psi, info): info = {"iterations", "omega", "N", "residual",
    "converged", "stopped", "diverged", "step" (the final step), "history"
    (iteration, omega, N, residual every ``callback_every`` iterations)}; psi is
    the last accepted state.
    """
    space = FieldSpace(model, t)
    grid = model.grid
    if tol is None:
        tol = _residual_tol(grid)
    ip = lambda a, b: float(_inner(grid, a, b))          # noqa: E731
    psi = space.real(psi if sector is None else sector(psi))
    cons = []
    for c in (deflate or []):                            # orthonormal (same inner product as ip)
        c = space.real(c.to(psi.device))
        for q in cons:
            c = c - ip(q, c) * q
        cons.append(c / math.sqrt(ip(c, c)))
    m = torch.tensor([max(abs(mu), 1.0)], dtype=torch.float64, device=psi.device)

    def project(f):
        if sector is not None:
            f = sector(f)
        for q in cons:
            f = f - ip(q, f) * q
        return f

    x = y = psi
    k_mom, streak, tau_cur, history, last = 0, 0, tau, [], None
    info = {"omega": math.nan, "N": math.nan, "residual": math.nan}
    stopped = converged = diverged = False
    for it in range(max_iter + 1):
        if it % 20 == 0:
            P, _ = space.preconditioner(y[None], m)
        E, h = space.energy_gradient(y[None])
        n = ip(y, y)
        omega = float(E[0]) - mu * n
        if not omega <= (math.inf if last is None else last["omega"] + 1e-11 * abs(last["omega"])):
            # Omega rose (or is not finite): back to the last accepted point, half the step, no momentum
            tau_cur *= 0.5
            if last is None or tau_cur < 1e-6 * tau:
                diverged = True
                break
            k_mom = streak = 0
            x = y = last["y"] - tau_cur * last["d"]
            continue
        g = project(h[0] - mu * y)
        res = math.sqrt(ip(g, g) / n) if n > 0 else 0.0
        info = {"omega": omega, "N": n, "residual": res}
        if it % callback_every == 0:
            history.append((it, omega, n, res))
            if callback is not None and callback(it, y, info):
                stopped = True
        d = project(P(g[None])[0])
        size = tau_cur * math.sqrt(ip(d, d) / n) if n > 0 else 0.0
        if size > 0.1:                                   # at most 10% of the field per step (runaway growth)
            d, k_mom = d * (0.1 / size), 0
        last = {"y": y, "d": d, "omega": omega}
        if stopped or res <= tol or it == max_iter:
            converged = res <= tol
            break
        streak += 1
        if streak >= 100 and tau_cur < tau:              # recover the step after 100 accepted iterations
            tau_cur, streak = min(2 * tau_cur, tau), 0
        x_new = y - tau_cur * d
        if ip(g, x_new - x) > 0:                         # gradient restart: momentum points uphill
            k_mom = 0
        k_mom += 1
        y = x_new + ((k_mom - 1) / (k_mom + 2)) * (x_new - x)
        x = x_new
    out = y if last is None else last["y"]
    return out, {"iterations": it, "omega": info["omega"], "N": info["N"], "residual": info["residual"],
                 "converged": converged, "stopped": stopped, "diverged": diverged, "step": tau_cur,
                 "history": history}
