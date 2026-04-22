"""Classical driven fluxonium utilities.

This module collects classical tools for a single fluxonium degree of freedom with
optional charge drive and flux drive.

State variables
---------------
phi : reduced flux / unwrapped phase coordinate (dimensionless, lives on R)
n   : conjugate charge-like momentum

Standard fluxonium Hamiltonian (default)
----------------------------------------
H(phi,n,t) = 4 EC n^2
             + EL/2 * [phi - phi_ext0 - A_flux cos(omega_d t)]^2
             - EJ cos(phi)
             + A_charge cos(omega_d t) * n

Important conventions
---------------------
- phi is treated as an unwrapped coordinate on the real line.
- For the undriven system, fixed points of the flow are supported.
- For the driven system, periodic points of the stroboscopic map are supported.
- The default Lyapunov routine follows the user's Julia workflow: first solve the
  trajectory on a fixed grid, then propagate the tangent matrix over each block with the
  Jacobian frozen at the saved trajectory point, followed by Gram–Schmidt.

Gauge choice
------------
By default, the external flux enters in the inductive term, which is the standard
fluxonium gauge. For static bias only, one can alternatively use the equivalent form

    H = 4 EC n^2 + EL/2 * phi^2 - EJ cos(phi - phi_ext0) + charge-drive

by setting gauge="cosine_static". This is NOT enabled for time-dependent flux drive
(A_flux != 0), because a time-dependent coordinate shift would generate extra terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root
from scipy.linalg import expm

ArrayLike = Sequence[float]
Gauge = Literal["inductive", "cosine_static"]


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
    gauge : {"inductive", "cosine_static"}, default "inductive"
        Where the static flux bias is placed.
        - "inductive": standard fluxonium gauge, EL/2 * (phi - phi_ext)^2 - EJ cos(phi)
        - "cosine_static": static-bias equivalent gauge, EL/2 * phi^2 - EJ cos(phi - phi_ext0)
          Only valid here when A_flux == 0.
    """

    EC: float
    EJ: float
    EL: float
    phi_ext0: float
    omega_d: float
    A_charge: float = 0.0
    A_flux: float = 0.0
    gauge: Gauge = "inductive"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def drive_period(p: FluxoniumParams) -> float:
    """Return the drive period 2π / omega_d."""
    return 2.0 * np.pi / p.omega_d


def wrap_to_pi(phi: np.ndarray | float) -> np.ndarray | float:
    """Wrap phase to (-π, π]. Useful for plotting only."""
    return (np.asarray(phi) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_to_center(phi: np.ndarray | float, center: float) -> np.ndarray | float:
    """Wrap phase to a 2π-window centered at `center`.

    This is often more informative than wrapping around zero when the interesting
    structure sits near phi ~ pi.
    """
    phi_arr = np.asarray(phi)
    return center + (phi_arr - center + np.pi) % (2.0 * np.pi) - np.pi


def phi_ext_t(t: np.ndarray | float, p: FluxoniumParams) -> np.ndarray | float:
    """Time-dependent reduced external flux."""
    return p.phi_ext0 + p.A_flux * np.cos(p.omega_d * np.asarray(t))


def _validate_params(p: FluxoniumParams) -> None:
    if p.gauge == "cosine_static" and abs(p.A_flux) > 0:
        raise ValueError(
            "gauge='cosine_static' is implemented only for static bias (A_flux = 0). "
            "For time-dependent flux drive, use the standard gauge='inductive'."
        )


# -----------------------------------------------------------------------------
# Potential and Hamiltonian
# -----------------------------------------------------------------------------

def potential(
    phi: np.ndarray | float,
    p: FluxoniumParams,
    *,
    phi_ext: Optional[float] = None,
) -> np.ndarray | float:
    """Undriven potential or frozen-time potential.

    gauge='inductive':
        U(phi) = EL/2 * (phi - phi_ext)^2 - EJ cos(phi)

    gauge='cosine_static' (static bias only):
        U(phi) = EL/2 * phi^2 - EJ cos(phi - phi_ext0)
    """
    _validate_params(p)
    phi_arr = np.asarray(phi)

    if p.gauge == "inductive":
        ext = p.phi_ext0 if phi_ext is None else phi_ext
        return 0.5 * p.EL * (phi_arr - ext) ** 2 - p.EJ * np.cos(phi_arr)

    ext = p.phi_ext0 if phi_ext is None else phi_ext
    return 0.5 * p.EL * phi_arr**2 - p.EJ * np.cos(phi_arr - ext)


def hamiltonian(t: float, u: ArrayLike, p: FluxoniumParams) -> float:
    """Full time-dependent Hamiltonian evaluated on a phase-space point."""
    _validate_params(p)
    phi, n = np.asarray(u, dtype=float)

    if p.gauge == "inductive":
        ext = phi_ext_t(t, p)
        return (
            4.0 * p.EC * n**2
            + 0.5 * p.EL * (phi - ext) ** 2
            - p.EJ * np.cos(phi)
            + p.A_charge * np.cos(p.omega_d * t) * n
        )

    return (
        4.0 * p.EC * n**2
        + 0.5 * p.EL * phi**2
        - p.EJ * np.cos(phi - p.phi_ext0)
        + p.A_charge * np.cos(p.omega_d * t) * n
    )


def dHdt_explicit(t: float, u: ArrayLike, p: FluxoniumParams) -> float:
    """Explicit time derivative ∂H/∂t evaluated along a trajectory."""
    _validate_params(p)
    phi, n = np.asarray(u, dtype=float)
    term_charge = -p.A_charge * p.omega_d * np.sin(p.omega_d * t) * n

    if p.gauge == "inductive":
        term_flux = p.EL * p.A_flux * p.omega_d * np.sin(p.omega_d * t) * (phi - phi_ext_t(t, p))
        return term_charge + term_flux

    return term_charge


# -----------------------------------------------------------------------------
# Equations of motion and Jacobian
# -----------------------------------------------------------------------------

def rhs(t: float, u: ArrayLike, p: FluxoniumParams) -> np.ndarray:
    """Classical equations of motion for the driven fluxonium."""
    _validate_params(p)
    phi, n = np.asarray(u, dtype=float)
    dphi = 8.0 * p.EC * n + p.A_charge * np.cos(p.omega_d * t)

    if p.gauge == "inductive":
        dn = -p.EL * (phi - phi_ext_t(t, p)) - p.EJ * np.sin(phi)
    else:
        dn = -p.EL * phi - p.EJ * np.sin(phi - p.phi_ext0)

    return np.array([dphi, dn], dtype=float)


def jacobian(t: float, u: ArrayLike, p: FluxoniumParams) -> np.ndarray:
    """Jacobian ∂f/∂u of the 2D state-space flow."""
    _validate_params(p)
    phi, _n = np.asarray(u, dtype=float)

    if p.gauge == "inductive":
        kappa = p.EL + p.EJ * np.cos(phi)
    else:
        kappa = p.EL + p.EJ * np.cos(phi - p.phi_ext0)

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
# Potential extrema
# -----------------------------------------------------------------------------

def find_potential_extrema(
    p: FluxoniumParams,
    *,
    phi_min: float = -4.0 * np.pi,
    phi_max: float = 4.0 * np.pi,
    nscan: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find minima and maxima of the undriven potential in the chosen gauge.

    Returns
    -------
    minima, maxima : arrays with columns [phi, U(phi)]
    """
    _validate_params(p)

    if p.gauge == "inductive":
        def dU(phi):
            return p.EL * (phi - p.phi_ext0) + p.EJ * np.sin(phi)

        def d2U(phi):
            return p.EL + p.EJ * np.cos(phi)
    else:
        def dU(phi):
            return p.EL * phi + p.EJ * np.sin(phi - p.phi_ext0)

        def d2U(phi):
            return p.EL + p.EJ * np.cos(phi - p.phi_ext0)

    grid = np.linspace(phi_min, phi_max, nscan)
    vals = dU(grid)

    roots: List[float] = []
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i + 1]
        fa, fb = vals[i], vals[i + 1]
        if fa == 0.0:
            roots.append(float(a))
        elif fa * fb < 0:
            roots.append(float(brentq(dU, float(a), float(b))))

    roots = sorted(roots)
    unique: List[float] = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > 1e-8:
            unique.append(r)

    minima, maxima = [], []
    for r in unique:
        if d2U(r) > 0:
            minima.append(r)
        elif d2U(r) < 0:
            maxima.append(r)

    minima = np.array(minima, dtype=float)
    maxima = np.array(maxima, dtype=float)

    return (
        np.column_stack((minima, potential(minima, p))) if minima.size else np.empty((0, 2)),
        np.column_stack((maxima, potential(maxima, p))) if maxima.size else np.empty((0, 2)),
    )


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


def solve_with_work(
    u0,
    t_span,
    p: FluxoniumParams,
    t_eval=None,
    method="DOP853",
    rtol=1e-6,
    atol=1e-6,
):
    """Solve the driven system together with the accumulated work variable W."""
    y0 = np.array([u0[0], u0[1], 0.0], dtype=float)

    def rhs_with_work(t, y, p_):
        phi, n, _w = y
        du = rhs(t, [phi, n], p_)
        dW = dHdt_explicit(t, [phi, n], p_)
        return np.array([du[0], du[1], dW], dtype=float)

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
    """Check H(t)-H(0)-W(t), where W was evolved by the same solver."""
    t = sol.t
    y = sol.y.T
    H = np.array([hamiltonian(tt, yy[:2], p) for tt, yy in zip(t, y)], dtype=float)
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
    wrap_center: Optional[float] = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> np.ndarray:
    """Return stroboscopic points for a collection of initial conditions."""
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
            center = 0.0 if wrap_center is None else float(wrap_center)
            pts[:, 0] = wrap_to_center(pts[:, 0], center=center)
        all_points.append(pts)

    return np.vstack(all_points) if all_points else np.empty((0, 2), dtype=float)


# -----------------------------------------------------------------------------
# Lyapunov routines
# -----------------------------------------------------------------------------

def GS(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Classical Gram–Schmidt, matching the user's Julia routine closely.

    Returns Q, R with positive diagonal entries when the column norm is nonzero.
    """
    A = np.array(A, dtype=float, copy=True)
    M, N = A.shape
    Q = np.zeros((M, N), dtype=float)
    R = np.zeros((N, N), dtype=float)

    for j in range(N):
        v = A[:, j].copy()
        for k in range(j):
            R[k, j] = np.dot(Q[:, k], v)
            v -= R[k, j] * Q[:, k]

        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            R[j, j] = norm_v
            Q[:, j] = v / norm_v
        else:
            R[j, j] = 0.0
            Q[:, j] = 0.0

    return Q, R


def G_evolution_frozenJ(t: float, g_flat: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Tangent-matrix evolution with frozen Jacobian J over one interval."""
    G = g_flat.reshape((2, 2))
    dG = J @ G
    return dG.ravel()


def lyapunov_max_julia_style(
    u_i: ArrayLike,
    p: FluxoniumParams,
    N: int,
    dt: float,
    err: float = 1e-10,
    transient_time: float = 100.0,
    method: str = "DOP853",
    traj_rtol: float = 1e-10,
    traj_atol: float = 1e-10,
    tangent_method: Literal["expm", "solve_ivp"] = "expm",
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Python version close to the user's Julia ``Lyapunov_max``.

    Workflow
    --------
    1. Solve the full trajectory first on a fixed grid.
    2. At each saved point x_i, compute J(x_i).
    3. Freeze J over one interval dt and evolve G via dG/dt = J G.
    4. Gram–Schmidt step.
    5. Accumulate log(diag(R)).

    Parameters
    ----------
    tangent_method : {"expm", "solve_ivp"}
        - "expm": exact propagation for frozen J, G(dt)=exp(J dt) G(0). Faster.
        - "solve_ivp": integrates dG/dt = JG over the block, closer to the original Julia logic.
    """
    u_i = np.asarray(u_i, dtype=float)
    t_eval = np.arange(0.0, N * dt, dt)

    prob = solve_ivp(
        fun=lambda t, u: rhs(t, u, p),
        t_span=(0.0, N * dt),
        y0=u_i,
        t_eval=t_eval,
        method=method,
        rtol=traj_rtol,
        atol=traj_atol,
    )

    if not prob.success:
        raise RuntimeError(f"Trajectory solve failed: {prob.message}")

    traj = prob.y.T
    G = np.eye(2, dtype=float)

    lam = np.zeros(2, dtype=float)
    lam1 = np.zeros(2, dtype=float)
    lam_t = np.zeros(N, dtype=float)

    for i in range(N):
        x = traj[i]
        t_i = i * dt
        J = jacobian(t_i, x, p)

        if tangent_method == "expm":
            G = expm(J * dt) @ G
        else:
            solG = solve_ivp(
                fun=lambda t, g: G_evolution_frozenJ(t, g, J),
                t_span=(0.0, dt),
                y0=G.ravel(),
                method=method,
                t_eval=[dt],
                rtol=err,
                atol=err,
            )
            if not solG.success:
                raise RuntimeError(f"Tangent solve failed at step {i}: {solG.message}")
            G = solG.y[:, -1].reshape((2, 2))

        G, R = GS(G)
        diagR = np.diag(R)
        diagR = np.where(np.abs(diagR) < 1e-300, 1e-300, diagR)

        if (i + 1) * dt > transient_time:
            lam += np.log(diagR)

        lam1 += np.log(diagR)
        lam_t[i] = np.max(lam1) / ((i + 1) * dt)

    T = (N - transient_time / dt) * dt
    if T <= 0:
        raise ValueError("transient_time is too large compared to N*dt")

    lam = lam / T
    lam_max = float(np.max(lam))
    return lam_max, lam_t, lam, traj


def lyapunov_until_converged(
    u0,
    p,
    *,
    dt=1e-2,
    max_time=1000.0,
    min_time=100.0,
    window_time=20.0,
    std_tol=1e-4,
    drift_tol=1e-4,
    consecutive_windows=3,
    method="DOP853",
    rtol=1e-10,
    atol=1e-10,
    tangent_method="expm",
):
    """
    Compute the largest Lyapunov exponent and stop automatically when the running
    estimate becomes stable over a time window.

    Stop rule:
      - std of lambda_max(t) over last window < std_tol
      - mean drift between consecutive windows < drift_tol
      - t >= min_time
      - condition satisfied for `consecutive_windows` consecutive checks
    """

    x = np.asarray(u0, dtype=float).copy()
    G = np.eye(2, dtype=float)

    t = 0.0
    lam_sum = np.zeros(2, dtype=float)

    times = []
    lam_max_t = []
    lam_local = []

    window_n = max(5, int(np.ceil(window_time / dt)))
    prev_window_mean = None
    stable_counter = 0

    while t < max_time:
        # Jacobian frozen at the beginning of the block, as in your Julia logic
        J = jacobian(t, x, p)

        # propagate tangent matrix over one block
        if tangent_method == "expm":
            G = expm(J * dt) @ G
        elif tangent_method == "solve_ivp":
            solG = solve_ivp(
                fun=lambda tt, g: G_evolution_frozenJ(tt, g, J),
                t_span=(0.0, dt),
                y0=G.ravel(),
                t_eval=[dt],
                method=method,
                rtol=rtol,
                atol=atol,
            )
            if not solG.success:
                raise RuntimeError(f"Tangent solve failed at t={t}: {solG.message}")
            G = solG.y[:, -1].reshape((2, 2))
        else:
            raise ValueError("tangent_method must be 'expm' or 'solve_ivp'")

        # GS step
        G, R = GS(G)

        diagR = np.diag(R)
        diagR = np.where(np.abs(diagR) < 1e-300, 1e-300, diagR)

        lam_step = np.log(diagR) / dt
        lam_local.append(lam_step.copy())

        lam_sum += np.log(diagR)

        # propagate the base trajectory to the next saved point
        solx = solve_ivp(
            fun=lambda tt, uu: rhs(tt, uu, p),
            t_span=(t, t + dt),
            y0=x,
            t_eval=[t + dt],
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not solx.success:
            raise RuntimeError(f"Trajectory solve failed at t={t}: {solx.message}")

        x = solx.y[:, -1]
        t += dt

        lam_running = np.max(lam_sum) / t
        times.append(t)
        lam_max_t.append(lam_running)

        # check convergence only after enough time and enough points
        if t >= min_time and len(lam_max_t) >= window_n:
            window = np.array(lam_max_t[-window_n:], dtype=float)
            window_mean = float(np.mean(window))
            window_std = float(np.std(window))

            if prev_window_mean is None:
                drift = np.inf
            else:
                drift = abs(window_mean - prev_window_mean)

            prev_window_mean = window_mean

            if (window_std < std_tol) and (drift < drift_tol):
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= consecutive_windows:
                break

    lam = lam_sum / t
    lam_max = float(np.max(lam))

    return {
        "lam": lam,
        "lam_max": lam_max,
        "times": np.asarray(times),
        "lam_max_t": np.asarray(lam_max_t),
        "lam_local": np.asarray(lam_local),
        "stop_time": t,
        "converged": stable_counter >= consecutive_windows,
        "window_n": window_n,
    }


# -----------------------------------------------------------------------------
# Undriven fixed points and their stability
# -----------------------------------------------------------------------------

def undriven_fixed_point_equation(phi: float, p: FluxoniumParams) -> float:
    """Equation defining undriven fixed points for A_charge = A_flux = 0."""
    _validate_params(p)
    if p.gauge == "inductive":
        return p.EL * (phi - p.phi_ext0) + p.EJ * np.sin(phi)
    return p.EL * phi + p.EJ * np.sin(phi - p.phi_ext0)


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
    """Classify linear stability of an undriven fixed point."""
    _validate_params(p)
    if p.gauge == "inductive":
        kappa = p.EL + p.EJ * np.cos(phi_star)
    else:
        kappa = p.EL + p.EJ * np.cos(phi_star - p.phi_ext0)

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
    "wrap_to_center",
    "phi_ext_t",
    "potential",
    "hamiltonian",
    "dHdt_explicit",
    "rhs",
    "jacobian",
    "state_tangent_rhs",
    "find_potential_extrema",
    "integrate_trajectory",
    "solve_with_work",
    "energy_balance_from_augmented_solution",
    "make_initial_conditions",
    "poincare_section",
    "GS",
    "G_evolution_frozenJ",
    "lyapunov_max_julia_style",
    "lyapunov_spectrum",
    "lyapunov_spectrum_fixed_blocks",
    "lyapunov_spectrum_rk4",
    "undriven_fixed_point_equation",
    "find_undriven_fixed_points",
    "classify_undriven_fixed_point",
    "stroboscopic_map",
    "poincare_residual",
    "find_periodic_point",
    "monodromy_matrix",
    "classify_periodic_orbit",
]
