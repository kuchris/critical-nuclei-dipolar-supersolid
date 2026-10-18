"""Bogoliubov-de Gennes excitations of a stationary state.

For a real stationary state psi0 (chemical potential mu) the linearized eGPE
for delta psi = u exp(-i w t) - v* exp(i w t) reduces, with f_pm = u +- v, to

    w f_- = L f_+,     w f_+ = M f_-,     so  L M f_- = w^2 f_-,

where L = H[psi0] - mu (the GPE operator; L psi0 = 0 is the gauge mode) and
M = L + 2 X with X f = psi0 dU[psi0 f], the linear response of the local terms
(contact, LHY, dipolar) to the density change psi0 (u - v). For a Bloch
quasi-momentum q along one axis, u, v = exp(i q x) * periodic and the kinetic
energy becomes |k + q|^2 / 2; every operator keeps q. With M Hermitian positive
definite (a stable state; at q = 0 a crystal has translation zero modes, so use
q != 0 there) the problem is the generalized Hermitian one

    L g = w^2 M^-1 g,         f_- = M^-1 g,

solved for the lowest eigenvalues by block LOBPCG on the GPU with the kinetic
preconditioner (|k + q|^2 / 2 + s)^-1; M^-1 is applied by preconditioned
conjugate gradients. (The equivalent (M L M) g = w^2 M g is third order and
converged ~100x slower.)

M is also the energy Hessian for real perturbations at fixed norm:
:func:`constrained_hessian` gives its lowest eigenvalues on the tangent space
of the sphere (the Morse index of a stationary state, e.g. a saddle point
found by :mod:`dipgpe.mep`), with the same LOBPCG.

The operators act in double precision for any field precision: a complex64
state is promoted (the complex64 dipolar kernels stay as they are), and the
LOBPCG blocks are complex128. In complex64 LOBPCG failed to converge from some
random starts (ISSUES C3-4).
"""

from __future__ import annotations

import math

import torch

from .diagnostics import chemical_potential, density


def _real_state(psi0):
    """psi0 with its global phase removed; it must then be real."""
    phase = torch.angle((psi0 * psi0).sum()) / 2
    psi = psi0 * torch.exp(-1j * phase)
    if float(psi.imag.abs().max()) > 1e-6 * float(psi.abs().max()):
        raise ValueError("bogoliubov needs a real stationary state (no rotation or current)")
    if float(psi.real.sum()) < 0:
        psi = -psi
    return psi.real.to(psi0.real.dtype)


class Bogoliubov:
    """Operators L, M (and the generalized problem) for one quasi-momentum q."""

    def __init__(self, model, psi0, q=0.0, axis=-1, t=0.0):
        grid = self.grid = model.grid
        for term in model.terms:
            if not hasattr(term, "local"):
                raise NotImplementedError(f"bogoliubov does not support {type(term).__name__}")
            if getattr(term, "nonlinear", True) and not hasattr(term, "response"):
                raise NotImplementedError(f"{type(term).__name__} has no linear response")
        self.model, self.q, self.axis, self.t = model, float(q), axis % grid.ndim, t
        self.psi0 = _real_state(psi0).to(torch.float64)
        self.rho0 = self.psi0 ** 2
        self.mu = float(chemical_potential(model, self.psi0.to(torch.complex128), t))
        self.U0 = model.local_potential(self.rho0, t)
        k2 = 0.0
        for i, ki in enumerate(grid.k64):
            k2 = k2 + (ki + (self.q if i == self.axis else 0.0)) ** 2
        self.kin = 0.5 * k2
        self.deflate = []

    def _fft(self, f):
        return torch.fft.fftn(f, dim=self.grid.dims)

    def _ifft(self, f):
        return torch.fft.ifftn(f, dim=self.grid.dims)

    def L(self, f):
        return self._ifft(self.kin * self._fft(f)) + (self.U0 - self.mu) * f

    def X(self, f):
        drho = self.psi0 * f
        dU = 0.0
        for term in self.model.local_terms:
            if getattr(term, "nonlinear", True):
                dU = dU + term.response(self.rho0, drho, self.q, self.axis)
        return self.psi0 * dU

    def M(self, f):
        return self.L(f) + 2 * self.X(f)

    def precondition(self, r, shift):
        return self._ifft(self._fft(r) / (self.kin + shift))

    def project(self, f):
        """Remove the components along the deflated fields (kernel of M, e.g. translation)."""
        for v in self.deflate:
            f = f - v * (v.conj() * f).sum(dim=self.grid.dims, keepdim=True)
        return f

    def solve_M(self, Y, shift, tol=1e-13, max_iter=5000):
        """M^-1 Y for a block Y (leading axis) by preconditioned conjugate gradients
        (within the complement of the deflated fields)."""
        dims = self.grid.dims
        dot = lambda a, b: (a.conj() * b).sum(dim=dims)
        expand = lambda c: c.reshape(c.shape + (1,) * self.grid.ndim)
        Y = self.project(Y)
        x = torch.zeros_like(Y)
        r = Y.clone()
        z = self.project(self.precondition(r, shift))
        p = z.clone()
        rz = dot(r, z)
        ynorm = torch.sqrt(dot(Y, Y).real)
        for _ in range(max_iter):
            Ap = self.M(p)
            alpha = rz / dot(p, Ap)
            x = x + expand(alpha) * p
            r = self.project(r - expand(alpha) * Ap)          # projected system P M P x = P Y
            if bool((torch.sqrt(dot(r, r).real) <= tol * ynorm).all()):
                return x
            z = self.project(self.precondition(r, shift))
            rz_new = dot(r, z)
            p = z + expand(rz_new / rz) * p
            rz = rz_new
        raise RuntimeError("bogoliubov: CG for M^-1 did not converge (is the state stable at this q?)")


def _inner(grid, a, b):
    """Gram matrix a^H b for blocks (m, *shape), (n, *shape) -> (m, n), times dV."""
    return (a.reshape(a.shape[0], -1).conj() @ b.reshape(b.shape[0], -1).T) * grid.dV


def _rayleigh_ritz(grid, S, AS, BS, m):
    """Lowest m Ritz pairs of (S^H A S, S^H B S), dropping near-dependent directions."""
    GA, GB = _inner(grid, S, AS), _inner(grid, S, BS)
    GA, GB = 0.5 * (GA + GA.conj().T), 0.5 * (GB + GB.conj().T)
    try:
        wB, VB = torch.linalg.eigh(GB)
    except torch.linalg.LinAlgError:                     # nearly dependent block: regularize the Gram matrix
        eps = 1e-13 * float(GB.diagonal().real.abs().max())
        wB, VB = torch.linalg.eigh(GB + eps * torch.eye(GB.shape[0], dtype=GB.dtype, device=GB.device))
    keep = wB > 1e-12 * wB.max()
    T = VB[:, keep] / torch.sqrt(wB[keep])
    w, Y = torch.linalg.eigh(T.conj().T @ GA @ T)
    C = T @ Y
    return w[:m], C[:, :m]


def bogoliubov(model, psi0, q=0.0, n_modes=6, axis=-1, tol=1e-6, max_iter=1000, block=None, seed=0,
               shift=None, x0=None, deflate=None, verbose=False):
    """Lowest Bogoliubov frequencies at quasi-momentum q (along ``axis``).

    Returns (omega, modes, info): omega ascending (float64 tensor, sqrt of the
    eigenvalues clipped at 0), modes the corresponding f_- = u - v (Bloch-periodic
    part), info = {"iterations", "residual", "mu", "eigenvalues"}. ``tol`` bounds
    the relative residual ||L g - w^2 M^-1 g|| / (w^2 ||M^-1 g||) of every wanted
    mode (modes with w ~ 0, e.g. the gauge mode, on the scale of the largest
    wanted w^2). ``x0`` (block of fields) warm-starts the iteration; note it is
    in the g = M f_- representation, i.e. pass ``info["g"]`` of a nearby q.
    ``deflate`` (fields) removes zero modes of M from the problem: at q = 0 a
    modulated state has the translation mode d psi0/dz in the kernel of M, which
    makes M^-1 singular; pass [d psi0/dz] to compute the other modes (the
    translation Goldstone pair is then absent from the spectrum).
    """
    op = Bogoliubov(model, psi0, q, axis)
    _add_deflation(op, deflate or [])
    m = block or n_modes + max(2, n_modes // 2)
    X = _start_block(op, m, seed, x0)
    shift = shift if shift is not None else max(abs(op.mu), 1.0)
    lam, X, AX, BX, res, it = _lobpcg(op.grid, X, op.L, lambda Y: op.solve_M(Y, shift), op, shift,
                                      n_modes, tol, max_iter, verbose=verbose)
    omega = torch.sqrt(torch.clamp(lam[:n_modes].real, min=0.0)).to(torch.float64)
    return omega, BX[:n_modes], {"iterations": it, "residual": res[:n_modes].tolist(), "mu": op.mu,
                                "eigenvalues": lam[:n_modes].tolist(), "g": X[:n_modes]}


def constrained_hessian(model, psi0, n_modes=3, tol=1e-6, max_iter=1000, block=None, seed=0, deflate=None,
                        x0=None, sector=None, mu=None, verbose=False):
    """Lowest eigenvalues of the energy Hessian at a real stationary state psi0 on
    the tangent space of the sphere ||psi||^2 = N, for real (amplitude) perturbations.

    That Hessian is M = L + 2X restricted to the complement of psi0: for a unit
    field f orthogonal to psi0 the energy on the sphere changes by eps^2 <f, M f>
    along psi0 + eps f. The number of negative eigenvalues is the Morse index (1 at
    a saddle point between two minima; Liu, Xie & Yuan, J. Comput. Phys. 473,
    111719 (2023), Def. 2.2). Phase perturbations (the operator L) are not
    included. ``deflate`` removes further zero modes (e.g. d psi0/dz in a periodic
    box). ``x0`` (fields) warm-starts the block, e.g. the path tangent at a saddle.
    ``sector`` (a linear projection commuting with M, e.g. onto fields even in z)
    restricts the problem to that symmetry sector.
    With ``mu`` (grand canonical, psi0 from ``refine_saddle(..., mu=mu)``) the
    Hessian of E - mu N on all real perturbations, psi0 included. Along a branch of
    stationary states M dpsi/dmu = psi0, so the two indices differ by one exactly
    when dN/dmu < 0 (Vakhitov-Kolokolov).
    Returns (eigenvalues (float64, ascending), modes, info) with info =
    {"iterations", "residual", "mu"}; ``tol`` bounds the relative residual.
    """
    op = Bogoliubov(model, psi0)
    if mu is not None:
        op.mu = mu
    _add_deflation(op, ([op.psi0] if mu is None else []) + list(deflate or []))
    if sector is not None:
        project = op.project
        op.project = lambda f: project(sector(f))
    m = block or n_modes + max(2, n_modes // 2)
    X = _start_block(op, m, seed, x0)
    shift = max(abs(op.mu), 1.0)
    apply_A = lambda Y: op.project(op.M(op.project(Y)))
    lam, X, _, _, res, it = _lobpcg(op.grid, X, apply_A, None, op, shift, n_modes, tol, max_iter,
                                    definite=False, verbose=verbose)
    return (lam[:n_modes].real.to(torch.float64), X[:n_modes],
            {"iterations": it, "residual": res[:n_modes].tolist(), "mu": op.mu})


def _start_block(op, m, seed, x0=None):
    """Random block weighted by |psi0| (optionally warm-started), projected off the deflated fields."""
    grid = op.grid
    gen = torch.Generator().manual_seed(seed)
    shape = (m,) + grid.shape
    X = torch.complex(torch.randn(shape, generator=gen, dtype=torch.float64),
                      torch.randn(shape, generator=gen, dtype=torch.float64)).to(grid.device)
    X = X * (op.psi0.abs() + 1e-3 * op.psi0.abs().max())
    if x0 is not None:                                   # warm start, e.g. modes of a nearby q
        k0 = min(len(x0), m)
        X[:k0] = x0[:k0].to(X.dtype)
    return op.project(X)


def _add_deflation(op, fields):
    for v in fields:                                     # orthonormalize (plain sum inner product)
        v = op.project(v.to(torch.complex128))
        op.deflate.append(v / torch.sqrt((v.conj() * v).sum().real))


def _lobpcg(grid, X, apply_A, apply_B, op, shift, n_modes, tol, max_iter, definite=True, verbose=False):
    """Lowest n_modes eigenpairs of A g = lam B g by block LOBPCG with the kinetic
    preconditioner of ``op`` (and its projection off the deflated fields), starting
    from the block X. ``apply_B`` None means B = 1 (then no B blocks are stored: the
    3D Hessian at 64 tube cells otherwise exceeded the GPU memory). With ``definite`` the problem is
    positive definite and a negative Ritz value (loss of positivity) restarts the
    iteration without the P block. Returns (lam, X, AX, BX, res, iterations)."""
    m = X.shape[0]
    ident = apply_B is None
    B = (lambda Y: Y) if ident else apply_B
    BX = B(X)
    AX = apply_A(X)
    lam, C = _rayleigh_ritz(grid, X, AX, BX, m)
    C = C.to(X.dtype)
    X, AX = (torch.einsum("ij,i...->j...", C, Z) for Z in (X, AX))
    BX = X if ident else torch.einsum("ij,i...->j...", C, BX)
    P = AP = BP = None
    res = None
    for it in range(1, max_iter + 1):
        R = AX - lam.reshape(-1, *([1] * grid.ndim)) * BX
        rn = torch.sqrt(_inner(grid, R, R).diagonal().real)
        bn = torch.sqrt(_inner(grid, BX, BX).diagonal().real)
        scale = torch.clamp(lam.real.abs(), min=float(lam[:n_modes].real.abs().max()) * 1e-3 + 1e-300)
        res = rn / (bn * scale)                          # relative residual; ~0 modes measured on the block scale
        if verbose and it % 20 == 0:
            print(it, lam[:n_modes].tolist(), float(res[:n_modes].max()))
        if bool((res[:n_modes] <= tol).all()):
            break
        active = res > tol                               # soft locking: no new directions for converged vectors
        W = op.project(op.precondition(R[active], shift))
        BW = W if ident else B(W)
        AW = apply_A(W)
        blocks = [(X, AX, BX), (W, AW, BW)]
        if P is not None:
            blocks.append((P[active], AP[active], P[active] if ident else BP[active]))
        S = torch.cat([b[0] for b in blocks])
        AS = torch.cat([b[1] for b in blocks])
        BS = S if ident else torch.cat([b[2] for b in blocks])
        del blocks, W, AW, BW
        d = (S.reshape(S.shape[0], -1).conj() * BS.reshape(S.shape[0], -1)).sum(1).real * grid.dV
        keep = d > 1e-14 * float(d[:m].max())            # drop new directions with ~0 (or negative) B-norm
        keep[:m] = True
        if not bool(keep.all()):
            S, AS, d = S[keep], AS[keep], d[keep]
            BS = S if ident else BS[keep]
        norms = torch.sqrt(d.clamp_min(1e-300))
        sc = (1 / norms).to(S.dtype).reshape(-1, *([1] * grid.ndim))
        S, AS = S * sc, AS * sc
        BS = S if ident else BS * sc
        try:
            lam_new, C = _rayleigh_ritz(grid, S, AS, BS, m)
        except torch.linalg.LinAlgError:
            if P is None:
                raise
            P = AP = BP = None                           # ill-conditioned with P: restart without it
            continue
        if definite and float(lam_new.real.min()) < -1e-8 * float(lam_new.real.abs().max()) and P is not None:
            P = AP = BP = None                           # loss of positivity: restart without P
            continue
        lam = lam_new
        C = C.to(S.dtype)
        Xn = torch.einsum("ij,i...->j...", C, S)
        AXn = torch.einsum("ij,i...->j...", C, AS)
        BXn = Xn if ident else torch.einsum("ij,i...->j...", C, BS)
        Cp = C.clone()
        Cp[:m] = 0                                       # the part of the update outside span(X)
        P = torch.einsum("ij,i...->j...", Cp, S)
        AP = torch.einsum("ij,i...->j...", Cp, AS)
        BP = P if ident else torch.einsum("ij,i...->j...", Cp, BS)
        X, AX, BX = Xn, AXn, BXn
        del S, AS, BS
    return lam, X, AX, BX, res, it
