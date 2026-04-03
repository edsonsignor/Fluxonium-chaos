from scipy.optimize import brentq

@dataclass
class FluxoniumDriveParams:
    EC: float                  # charging energy (angular units)
    EJ: float                  # Josephson energy (angular units)
    EL: float                  # inductive energy (angular units)
    phi_ext0: float            # static reduced flux bias
    omega_d: float             # drive angular frequency
    drive_mode: str = "flux"   # "flux" or "charge"
    A_flux: float = 0.0        # flux-drive amplitude in reduced phase units
    A_n: float = 0.0           # charge-drive amplitude in angular units

def wrap_to_pi(phi):
    """Wrap phase to (-pi, pi]."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi

def phi_ext_of_t(t, p: FluxoniumDriveParams):
    """Time-dependent reduced external flux."""
    if p.drive_mode == "flux":
        return p.phi_ext0 + p.A_flux * np.cos(p.omega_d * t)
    return p.phi_ext0

def rhs(t, y, p: FluxoniumDriveParams):
    """
    Classical equations of motion for driven fluxonium.
    y = [phi, n]
    """
    phi, n = y

    # phi dot
    phi_dot = 8.0 * p.EC * n
    if p.drive_mode == "charge":
        phi_dot += p.A_n * np.cos(p.omega_d * t)

    # n dot
    n_dot = -p.EL * (phi - phi_ext_of_t(t, p)) - p.EJ * np.sin(phi)

    return np.array([phi_dot, n_dot], dtype=float)

def make_initial_conditions(phi_min, phi_max, n_min, n_max,
                            n_phi=8, n_n=8, random=False, seed=0):
    """
    Build a set of initial conditions in a rectangular box.
    """
    rng = np.random.default_rng(seed)

    if random:
        n_ic = n_phi * n_n
        phi0 = rng.uniform(phi_min, phi_max, n_ic)
        n0 = rng.uniform(n_min, n_max, n_ic)
        return np.column_stack([phi0, n0])

    phi_vals = np.linspace(phi_min, phi_max, n_phi)
    n_vals = np.linspace(n_min, n_max, n_n)
    PHI, N = np.meshgrid(phi_vals, n_vals, indexing="xy")
    return np.column_stack([PHI.ravel(), N.ravel()])

def poincare_section(params: FluxoniumDriveParams,
                     initial_conditions,
                     n_transient=1000,
                     n_strobes=2000,
                     phase_fraction=0.0,
                     wrap_phi_for_plot=False,
                     method="DOP853",
                     rtol=1e-10,
                     atol=1e-12):
    """
    Integrate many trajectories and sample them stroboscopically:
        t_k = phase_fraction*T + k*T
    after discarding an initial transient.
    """
    T = 2.0 * np.pi / params.omega_d

    # Stroboscopic times
    t_eval = phase_fraction * T + T * np.arange(n_transient + n_strobes)
    t_final = t_eval[-1]

    all_points = []

    for y0 in np.asarray(initial_conditions):
        sol = solve_ivp(
            fun=lambda t, y: rhs(t, y, params),
            t_span=(0.0, t_final),
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        phi = sol.y[0, n_transient:]
        n = sol.y[1, n_transient:]

        if wrap_phi_for_plot:
            phi = wrap_to_pi(phi)

        all_points.append(np.column_stack([phi, n]))

    return np.vstack(all_points)

def plot_poincare(points, wrapped=False, title=None, s=0.2, alpha=0.7):
    plt.figure(figsize=(7.0, 5.2))
    plt.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, rasterized=True)
    plt.xlabel(r"$\phi$" + (r" mod $2\pi$" if wrapped else ""))
    plt.ylabel(r"$n$")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()
 


# -------------------------------------------------------------------
# Example
# -------------------------------------------------------------------
if __name__ == "__main__":
    GHz = 2.0 * np.pi  # multiply a value in GHz by this

    # Example parameters: charge drive near half flux
    params = FluxoniumDriveParams(
        EC=1.0 * GHz,          # ~1 GHz
        EJ=8.5 * GHz,          # ~8.5 GHz
        EL=0.5 * GHz,          # ~0.5 GHz
        phi_ext0 = np.pi,        # half-flux bias
        omega_d= 2.0 * GHz,     # drive frequency
        drive_mode="charge",    # charge drive
        A_flux=0.0,            # drive amplitude in reduced phase units
        A_n=0.9,
    )

    # Initial conditions:
    # unwrapped phi is often better for fluxonium because of the inductive parabola
    ics = make_initial_conditions(
        phi_min=-np.pi,
        phi_max= np.pi,
        n_min=-8.0,
        n_max= 8.0,
        n_phi=10,
        n_n=8,
        random=True,
        seed=4,
    )

    """
    # Poincare section at phase_fraction = 0.0
    points = poincare_section(
        params=params,
        initial_conditions=ics,
        n_transient=0,
        n_strobes=200,
        phase_fraction=0.0,      # use 1/8 if you want a shifted strobe phase
        wrap_phi_for_plot=True, # keep unwrapped to see interwell transport
    )

    plot_poincare(
        points,
        wrapped=True,
        title="Driven fluxonium: stroboscopic Poincare section"
    )
    """
    # Also useful: wrapped phase plot
    points_wrapped = poincare_section(
        params=params,
        initial_conditions=ics,
        n_transient= 0,
        n_strobes=200,
        phase_fraction=0.0,
        wrap_phi_for_plot=True,
    )

    plot_poincare(
        points_wrapped,
        wrapped=True,
        title="Driven fluxonium: wrapped phase Poincare section"
    )


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


def plot_fluxonium_potential_with_extrema(
    EJ,
    EL,
    phi_ext,
    phi_min=-4*np.pi,
    phi_max=4*np.pi,
    npoints=4000,
    subtract_min=False,
):
    phi, U, ax = plot_fluxonium_potential(
        EJ, EL, phi_ext,
        phi_min=phi_min,
        phi_max=phi_max,
        npoints=npoints,
        subtract_min=subtract_min,
    )

    minima, maxima = find_potential_extrema(EJ, EL, phi_ext, phi_min, phi_max)

    def Ufun(x):
        val = 0.5 * EL * (x - phi_ext)**2 - EJ * np.cos(x)
        if subtract_min:
            val -= np.min(U)
        return val

    if len(minima) > 0:
        ax.plot(minima, [Ufun(x) for x in minima], 'o', label='minima')

    if len(maxima) > 0:
        ax.plot(maxima, [Ufun(x) for x in maxima], 's', label='maxima')

    ax.legend()
    return phi, U, minima, maxima, ax