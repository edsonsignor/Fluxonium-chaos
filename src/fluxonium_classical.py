"""Classical driven fluxonium utilities.

This module collects the classical tools for a single
fluxonium degree of freedom with optional charge drive and flux drive.

State variables
---------------
phi : reduced flux / unwrapped phase coordinate (dimensionless, lives on R)
n   : conjugate charge-like momentum

Hamiltonian
-----------
H(phi,n,t) = 4 EC n^2
             + EL/2 * [phi - phi_ext0 - A_flux cos(omega_d t)]^2
             - EJ cos(phi)
             + A_charge cos(omega_d t) * n

Conventions
-----------
- phi is treated as an unwrapped coordinate on the real line.
- For the undriven system, fixed points of the flow are supported.
- For the driven system, periodic points of the stroboscopic map are supported.
- Lyapunov exponents are computed by evolving the state and tangent matrix
  simultaneously, with QR reorthonormalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root
from typing import Dict


ArrayLike = Sequence[float]


@dataclass
class FluxoniumParams:
    """Parameters for the classical driven fluxonium model.

    Parameters
    ----------
    EC, EJ, EL : float
        Circuit energies in consistent units.
    phi_ext0 : float
        Static reduced external flux.
    omega_d : float
        Drive angular frequency.
    A_charge : float, default 0.0
        Charge-drive amplitude multiplying n * cos(omega_d t).
    A_flux : float, default 0.0
        Flux-drive amplitude entering phi_ext(t) = phi_ext0 + A_flux cos(omega_d t).
    """

    EC: float
    EJ: float
    EL: float
    phi_ext0: float
    omega_d: float
    A_charge: float = 0.0
    A_flux: float = 0.0


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def drive_period(p: FluxoniumParams) -> float:
    """Return the drive period 2π / omega_d."""
    return 2.0 * np.pi / p.omega_d


def wrap_to_pi(phi: np.ndarray | float) -> np.ndarray | float:
    """Wrap phase to (-π, π]. Useful for plotting only."""
    return (np.asarray(phi) + np.pi) % (2.0 * np.pi) - np.pi


def phi_ext_t(t: np.ndarray | float, p: FluxoniumParams) -> np.ndarray | float:
    """Time-dependent reduced external flux."""
    return p.phi_ext0 + p.A_flux * np.cos(p.omega_d * np.asarray(t))


# -----------------------------------------------------------------------------
# Undriven Potential and Fixed points
# -----------------------------------------------------------------------------

def potential(phi: np.ndarray | float, p: FluxoniumParams, *, phi_ext: Optional[float] = None) -> np.ndarray | float:
    """Undriven fluxonium potential or frozen-time driven potential.

    U(phi) = EL/2 * (phi - phi_ext)^2 - EJ cos(phi)

    Parameters
    ----------
    phi : float or array
        Coordinate value(s).
    p : FluxoniumParams
        Model parameters.
    phi_ext : float, optional
        External reduced flux used in the potential. If omitted, use p.phi_ext0.
    """
    ext = p.phi_ext0 if phi_ext is None else phi_ext
    phi_arr = np.asarray(phi)
    return 0.5 * p.EL * (phi_arr - ext) ** 2 - p.EJ * np.cos(phi_arr)

def find_potential_extrema(EJ, EL, phi_ext, phi_min=-np.pi, phi_max=np.pi, nscan=10000):
    """
    Find extrema of U(phi) by solving U'(phi)=0:
        U'(phi) = EL (phi - phi_ext) + EJ sin(phi)
    """
    def dU(phi):
        return EL * (phi - phi_ext) + EJ * np.sin(phi)

    def d2U(phi):
        return EL + EJ * np.cos(phi)

    grid = np.linspace(phi_min, phi_max, nscan)
    vals = dU(grid)

    roots = []
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i+1]
        fa, fb = vals[i], vals[i+1]

        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0:
            roots.append(brentq(dU, a, b))

    # remove duplicates
    roots = sorted(roots)
    unique = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > 1e-8:
            unique.append(r)

    minima = []
    maxima = []
    for r in unique:
        if d2U(r) > 0:
            minima.append(r)
        elif d2U(r) < 0:
            maxima.append(r)

    return np.array(minima), np.array(maxima)


# -----------------------------------------------------------------------------
# Driven Hamiltonian
# -----------------------------------------------------------------------------

def hamiltonian(t: float, u: ArrayLike, p: FluxoniumParams) -> float:
    """Full time-dependent Hamiltonian evaluated on a phase-space point."""
    phi, n = np.asarray(u, dtype=float)
    ext = phi_ext_t(t, p)
    return (
        4.0 * p.EC * n**2
        + 0.5 * p.EL * (phi - ext) ** 2
        - p.EJ * np.cos(phi)
        + p.A_charge * np.cos(p.omega_d * t) * n
    )


def dHdt_explicit(t: float, u: ArrayLike, p: FluxoniumParams) -> float:
    """Explicit time derivative ∂H/∂t evaluated along a trajectory.

    This is useful for energy-balance diagnostics in the driven system.
    """
    phi, n = np.asarray(u, dtype=float)
    term_charge = -p.A_charge * p.omega_d * np.sin(p.omega_d * t) * n
    term_flux = p.EL * p.A_flux * p.omega_d * np.sin(p.omega_d * t) * (phi - phi_ext_t(t, p))
    return term_charge + term_flux


# -----------------------------------------------------------------------------
# Equations of motion and Jacobian
# -----------------------------------------------------------------------------

def rhs(t: float, u: ArrayLike, p: FluxoniumParams) -> np.ndarray:
    """Classical equations of motion for the driven fluxonium."""
    phi, n = np.asarray(u, dtype=float)
    dphi = 8.0 * p.EC * n + p.A_charge * np.cos(p.omega_d * t)
    dn = -p.EL * (phi - phi_ext_t(t, p)) - p.EJ * np.sin(phi)
    return np.array([dphi, dn], dtype=float)


def jacobian(t: float, u: ArrayLike, p: FluxoniumParams) -> np.ndarray:
    """Jacobian ∂f/∂u of the 2D state-space flow.

    The explicit t dependence enters only through the trajectory, not directly in
    the matrix elements.
    """
    phi, _n = np.asarray(u, dtype=float)
    kappa = p.EL + p.EJ * np.cos(phi)
    return np.array([[0.0, 8.0 * p.EC], [-kappa, 0.0]], dtype=float)


def state_tangent_rhs(t: float, y: np.ndarray, p: FluxoniumParams) -> np.ndarray:
    """Combined evolution of state and tangent matrix.

    y = [phi, n, G11, G12, G21, G22], where G is the 2x2 tangent matrix.
    """
    phi, n = y[0], y[1]
    g = y[2:].reshape(2, 2)

    du = rhs(t, [phi, n], p)
    dg = jacobian(t, [phi, n], p) @ g

    out = np.empty_like(y)
    out[:2] = du
    out[2:] = dg.ravel()
    return out


# -----------------------------------------------------------------------------
# Integration helpers
# -----------------------------------------------------------------------------

def integrate_trajectory(
    u0: ArrayLike,
    p: FluxoniumParams,
    t_span: Tuple[float, float],
    *,
    t_eval: Optional[np.ndarray] = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
):
    """Integrate the classical trajectory and return the scipy solution object."""
    u0_arr = np.asarray(u0, dtype=float)
    sol = solve_ivp(
        fun=lambda tt, uu: rhs(tt, uu, p),
        t_span=t_span,
        y0=u0_arr,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def solve_with_work(u0, t_span, p: FluxoniumParams, t_eval=None,
                    method="DOP853", rtol=1e-6, atol=1e-6):
    """
    Solve the driven fluxonium together with the accumulated work variable W.
    """
    y0 = np.array([u0[0], u0[1], 0.0], dtype=float)

    def rhs_with_work(t, y, p: FluxoniumParams):
        """
        y = [phi, n, W]
        where W satisfies dW/dt = ∂H/∂t
        """
        phi, n, W = y

        dphi = 8.0 * p.EC * n + p.A_charge * np.cos(p.omega_d * t)
        dn = -p.EL * (phi - phi_ext_t(t, p)) - p.EJ * np.sin(phi)
        dW = dHdt_explicit(t, y[:2], p)

        return np.array([dphi, dn, dW], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rhs_with_work(t, y, p),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return sol


def energy_balance_from_augmented_solution(sol, p: FluxoniumParams) -> Dict[str, np.ndarray | float]:
    """
    Check H(t)-H(0)-W(t), where W was evolved by the same solver.
    """
    t = sol.t
    y = sol.y.T

    H = np.array([hamiltonian(tt, yy, p) for tt, yy in zip(t, y[:,0:2])], dtype=float)
    W = y[:, 2]

    delta_H = H - H[0]
    err = delta_H - W

    return {
        "t": t,
        "H": H,
        "delta_H": delta_H,
        "work_solver": W,
        "error": err,
        "max_abs_error": float(np.max(np.abs(err))),
        "rms_error": float(np.sqrt(np.mean(err**2))),
    }

# -----------------------------------------------------------------------------
# Poincare section
# -----------------------------------------------------------------------------

def make_initial_conditions(
    phi_min: float,
    phi_max: float,
    n_min: float,
    n_max: float,
    *,
    n_phi: int = 8,
    n_n: int = 8,
    random: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Create a collection of initial conditions in a rectangle of phase space."""
    rng = np.random.default_rng(seed)
    if random:
        count = n_phi * n_n
        phi0 = rng.uniform(phi_min, phi_max, count)
        n0 = rng.uniform(n_min, n_max, count)
        return np.column_stack([phi0, n0])

    phi_vals = np.linspace(phi_min, phi_max, n_phi)
    n_vals = np.linspace(n_min, n_max, n_n)
    phi_grid, n_grid = np.meshgrid(phi_vals, n_vals, indexing="xy")
    return np.column_stack([phi_grid.ravel(), n_grid.ravel()])


def poincare_section(
    p: FluxoniumParams,
    initial_conditions: np.ndarray,
    *,
    n_discard: int = 500,
    n_strobes: int = 2000,
    phase_fraction: float = 0.0,
    wrap_phi_for_plot: bool = False,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> np.ndarray:
    """Return stroboscopic points for a collection of initial conditions.

    Notes
    -----
    `n_discard` is a plotting convenience for conservative Hamiltonian dynamics.
    It discards early stroboscopic points so the figure is less dominated by the
    arbitrary launch phase and initial placement.
    """
    t_period = drive_period(p)
    t_eval = phase_fraction * t_period + t_period * np.arange(n_discard + n_strobes)
    t_final = float(t_eval[-1])

    all_points: List[np.ndarray] = []
    for u0 in np.asarray(initial_conditions, dtype=float):
        sol = integrate_trajectory(
            u0=u0,
            p=p,
            t_span=(0.0, t_final),
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        pts = sol.y[:, n_discard:].T.copy()
        if wrap_phi_for_plot:
            pts[:, 0] = wrap_to_pi(pts[:, 0])
        all_points.append(pts)

    return np.vstack(all_points) if all_points else np.empty((0, 2), dtype=float)


# -----------------------------------------------------------------------------
# Lyapunov spectrum
# -----------------------------------------------------------------------------

def lyapunov_spectrum(
    u0: ArrayLike,
    p: FluxoniumParams,
    *,
    t_trans: float = 100.0,
    t_total: float = 2000.0,
    delta_r: Optional[float] = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute the 2D Lyapunov spectrum and running largest exponent.

    The state and tangent matrix are evolved simultaneously. QR
    reorthonormalization is performed every `delta_r`. For a periodically driven
    system, using one drive period is a natural choice.
    """
    if delta_r is None:
        delta_r = drive_period(p)

    y = np.zeros(6, dtype=float)
    y[:2] = np.asarray(u0, dtype=float)
    y[2:] = np.eye(2).ravel()

    t = 0.0
    t_end = t_trans + t_total
    n_steps = int(np.ceil(t_end / delta_r))

    lam_sum = np.zeros(2, dtype=float)
    elapsed = 0.0
    times: List[float] = []
    lam_max_t: List[float] = []

    for _ in range(n_steps):
        sol = solve_ivp(
            fun=lambda tt, yy: state_tangent_rhs(tt, yy, p),
            t_span=(t, t + delta_r),
            y0=y,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        y = sol.y[:, -1].copy()
        t = float(sol.t[-1])

        g = y[2:].reshape(2, 2)
        q, r = np.linalg.qr(g)

        # keep diagonal positive when possible
        for j in range(2):
            if r[j, j] < 0:
                r[j, :] *= -1.0
                q[:, j] *= -1.0

        y[2:] = q.ravel()

        if t > t_trans:
            lam_sum += np.log(np.abs(np.diag(r)))
            elapsed += delta_r
            times.append(t)
            lam_max_t.append(float(np.max(lam_sum) / elapsed))

    lam = lam_sum / elapsed
    lam_max = float(np.max(lam))
    return lam, lam_max, np.asarray(times), np.asarray(lam_max_t)


# -----------------------------------------------------------------------------
# Undriven fixed points and their stability
# -----------------------------------------------------------------------------

def undriven_fixed_point_equation(phi: float, p: FluxoniumParams) -> float:
    """Equation defining undriven fixed points for A_charge = A_flux = 0.

    n* = 0 and EL(phi* - phi_ext0) + EJ sin(phi*) = 0.
    """
    return p.EL * (phi - p.phi_ext0) + p.EJ * np.sin(phi)


def find_undriven_fixed_points(
    p: FluxoniumParams,
    *,
    phi_min: float = -6.0 * np.pi,
    phi_max: float = 6.0 * np.pi,
    n_scan: int = 20000,
) -> List[Dict[str, float]]:
    """Find fixed points of the undriven flow by scanning for sign changes."""
    if abs(p.A_charge) > 0 or abs(p.A_flux) > 0:
        raise ValueError("Set A_charge = A_flux = 0 to find true undriven fixed points.")

    grid = np.linspace(phi_min, phi_max, n_scan)
    vals = undriven_fixed_point_equation(grid, p)
    roots: List[float] = []

    for i in range(len(grid) - 1):
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = float(vals[i]), float(vals[i + 1])
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0:
            roots.append(float(brentq(lambda x: undriven_fixed_point_equation(x, p), a, b)))

    roots = sorted(roots)
    unique_roots: List[float] = []
    for r in roots:
        if not unique_roots or abs(r - unique_roots[-1]) > 1e-8:
            unique_roots.append(r)

    return [{"phi": r, "n": 0.0} for r in unique_roots]


def classify_undriven_fixed_point(phi_star: float, p: FluxoniumParams, *, tol: float = 1e-12) -> Dict[str, object]:
    """Classify linear stability of an undriven fixed point.

    J* = [[0, 8 EC], [-(EL + EJ cos(phi*)), 0]]
    """
    kappa = p.EL + p.EJ * np.cos(phi_star)
    if kappa > tol:
        omega_loc = np.sqrt(8.0 * p.EC * kappa)
        eigvals = np.array([1j * omega_loc, -1j * omega_loc])
        return {"type": "elliptic", "K": kappa, "eigenvalues": eigvals, "omega_local": float(omega_loc)}

    if kappa < -tol:
        growth = np.sqrt(8.0 * p.EC * (-kappa))
        eigvals = np.array([growth, -growth])
        return {"type": "hyperbolic", "K": kappa, "eigenvalues": eigvals, "growth_rate": float(growth)}

    return {"type": "parabolic", "K": kappa, "eigenvalues": np.array([0.0, 0.0])}


# -----------------------------------------------------------------------------
# Periodic points of the stroboscopic map
# -----------------------------------------------------------------------------

def stroboscopic_map(
    u0: ArrayLike,
    p: FluxoniumParams,
    *,
    n_periods: int = 1,
    phase_fraction: float = 0.0,
    method: str = "DOP853",
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> np.ndarray:
    """Return the phase-space point after `n_periods` of the drive."""
    t_period = drive_period(p)
    t0 = phase_fraction * t_period
    tf = t0 + n_periods * t_period
    sol = integrate_trajectory(
        u0=u0,
        p=p,
        t_span=(t0, tf),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return sol.y[:, -1].copy()


def poincare_residual(
    u: ArrayLike,
    p: FluxoniumParams,
    *,
    n_periods: int = 1,
    phase_fraction: float = 0.0,
) -> np.ndarray:
    """Residual F(u)-u for the stroboscopic map."""
    u_arr = np.asarray(u, dtype=float)
    return stroboscopic_map(u_arr, p, n_periods=n_periods, phase_fraction=phase_fraction) - u_arr


def find_periodic_point(
    u_guess: ArrayLike,
    p: FluxoniumParams,
    *,
    n_periods: int = 1,
    phase_fraction: float = 0.0,
):
    """Find a period-n_periods point of the stroboscopic map."""
    guess = np.asarray(u_guess, dtype=float)
    sol = root(lambda u: poincare_residual(u, p, n_periods=n_periods, phase_fraction=phase_fraction), guess)
    return sol.x, sol


def monodromy_matrix(
    u_star: ArrayLike,
    p: FluxoniumParams,
    *,
    n_periods: int = 1,
    phase_fraction: float = 0.0,
    method: str = "DOP853",
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> np.ndarray:
    """Return the tangent map over `n_periods` of the drive."""
    t_period = drive_period(p)
    t0 = phase_fraction * t_period
    tf = t0 + n_periods * t_period

    y0 = np.zeros(6, dtype=float)
    y0[:2] = np.asarray(u_star, dtype=float)
    y0[2:] = np.eye(2).ravel()

    sol = solve_ivp(
        fun=lambda tt, yy: state_tangent_rhs(tt, yy, p),
        t_span=(t0, tf),
        y0=y0,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    yf = sol.y[:, -1]
    return yf[2:].reshape(2, 2).copy()


def classify_periodic_orbit(
    u_star: ArrayLike,
    p: FluxoniumParams,
    *,
    n_periods: int = 1,
    phase_fraction: float = 0.0,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """Classify a periodic point through the multipliers of the Poincare map."""
    m = monodromy_matrix(u_star, p, n_periods=n_periods, phase_fraction=phase_fraction)
    multipliers = np.linalg.eigvals(m)
    det_m = float(np.linalg.det(m))
    tr_m = float(np.trace(m))

    if np.all(np.isreal(multipliers)) and np.max(np.abs(multipliers)) > 1.0 + tol:
        orbit_type = "hyperbolic"
    elif np.all(np.abs(np.abs(multipliers) - 1.0) < 1e-4):
        orbit_type = "elliptic_or_neutral"
    else:
        orbit_type = "near_resonant_or_unclear"

    return {
        "type": orbit_type,
        "M": m,
        "multipliers": multipliers,
        "detM": det_m,
        "trM": tr_m,
    }


__all__ = [
    "FluxoniumParams",
    "drive_period",
    "wrap_to_pi",
    "phi_ext_t",
    "potential",
    "find_potential_extrema",
    "hamiltonian",
    "dHdt_explicit",
    "rhs",
    "jacobian",
    "state_tangent_rhs",
    "integrate_trajectory",
    "solve_with_work",
    "energy_balance_from_augmented_solution",
    "make_initial_conditions",
    "poincare_section",
    "lyapunov_spectrum",
    "undriven_fixed_point_equation",
    "find_undriven_fixed_points",
    "classify_undriven_fixed_point",
    "stroboscopic_map",
    "poincare_residual",
    "find_periodic_point",
    "monodromy_matrix",
    "classify_periodic_orbit",
]
