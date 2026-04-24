import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import numpy as np

# -------------------------------------------------------------------
# Units:
# Use energies/frequencies in ANGULAR units.
# Convenient choice: time in ns, energies in rad/ns.
# Then 1 GHz corresponds to 2*pi rad/ns.
# -------------------------------------------------------------------
GHz = 2.0 * np.pi  # multiply a value in GHz by this

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fluxonium_classical import FluxoniumParams, lyapunov_until_converged, make_initial_conditions

EC, EJ, EL = 1.0, 3.6, 0.7
phi_ext0 = 2*np.pi*0.5
A_flux =0.
sq_num_ics = 2 # sq_num_ics^2 is the number of ICs


N_density = 3
omega_ds = np.linspace(0., 5., N_density)
A_charges = np.linspace(0., 5., N_density)

outdir = REPO_ROOT / "data" / "lyapunov_maps"
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / "lyapunov_map.h5"

# Paremters for the Lyapunov calculation
dt=1e-2
max_time=1000.0
min_time=100.0
window_time=20.0
std_tol=1e-2
drift_tol=1e-1
consecutive_windows=3
tangent_method="expm"


def parse_args():
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Compute a driven-fluxonium Lyapunov map.")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=outfile,
        help="Output HDF5 file.",
    )
    return parser.parse_args()


def save_h5(path, lambda_matrix, stop_time_matrix, converged_matrix):
    tmp_path = path.with_suffix(".tmp.h5")
    with h5py.File(tmp_path, "w") as h5:
        h5.create_dataset("omega_ds", data=omega_ds)
        h5.create_dataset("A_charges", data=A_charges)
        h5.create_dataset("lambda_matrix", data=lambda_matrix)
        h5.create_dataset("stop_time_matrix", data=stop_time_matrix)
        h5.create_dataset("converged_matrix", data=converged_matrix)

        h5.attrs["EC"] = EC
        h5.attrs["EJ"] = EJ
        h5.attrs["EL"] = EL
        h5.attrs["phi_ext0"] = phi_ext0
        h5.attrs["A_flux"] = A_flux
        h5.attrs["sq_num_ics"] = sq_num_ics
        h5.attrs["N_density"] = N_density
        h5.attrs["dt"] = dt
        h5.attrs["max_time"] = max_time
        h5.attrs["min_time"] = min_time
        h5.attrs["window_time"] = window_time
        h5.attrs["std_tol"] = std_tol
        h5.attrs["drift_tol"] = drift_tol
        h5.attrs["consecutive_windows"] = consecutive_windows
        h5.attrs["tangent_method"] = tangent_method
    tmp_path.replace(path)


def compute_grid_point(task):
    i, j, omega_d, A_charge = task
    param = FluxoniumParams(
        EC=EC,
        EJ=EJ,
        EL=EL,
        phi_ext0=phi_ext0,
        omega_d=omega_d,
        A_charge=A_charge,
        A_flux=A_flux,
    )

    ics = make_initial_conditions(
        phi_min=-2*np.pi,
        phi_max=2*np.pi,
        n_min=-2.0,
        n_max=2.0,
        n_phi=sq_num_ics,
        n_n=sq_num_ics,
        # random=True,
        # seed=4,
    )

    lambda_per_p = 0.0
    stop_time_per_p = 0.0
    converged_count = 0
    for u0 in ics:
        result = lyapunov_until_converged(
            u0=np.asarray(u0, dtype=float),
            p=param,
            dt=dt,
            max_time=max_time,
            min_time=min_time,
            window_time=window_time,
            std_tol=std_tol,
            drift_tol=drift_tol,
            consecutive_windows=consecutive_windows,
            tangent_method=tangent_method,
        )
        lambda_per_p += result["lam_max"]
        stop_time_per_p += result["stop_time"]
        converged_count += int(result["converged"])

    n_ics = len(ics)
    return (
        i,
        j,
        float(lambda_per_p / n_ics),
        float(stop_time_per_p / n_ics),
        converged_count == n_ics,
    )


def main():
    args = parse_args()
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    lambda_matrix = np.full((N_density, N_density), np.nan, dtype=float)
    stop_time_matrix = np.full((N_density, N_density), np.nan, dtype=float)
    converged_matrix = np.zeros((N_density, N_density), dtype=bool)

    tasks = [
        (i, j, float(omega_d), float(A_charge))
        for i, omega_d in enumerate(omega_ds)
        for j, A_charge in enumerate(A_charges)
    ]

    workers = max(1, min(args.workers, len(tasks)))
    print(f"Running {len(tasks)} grid points with {workers} worker(s).", flush=True)

    if workers == 1:
        completed = 0
        for task in tasks:
            i, j, lam, stop_time, converged = compute_grid_point(task)
            lambda_matrix[i, j] = lam
            stop_time_matrix[i, j] = stop_time
            converged_matrix[i, j] = converged
            completed += 1
            print(
                f"Finished {completed}/{len(tasks)}: "
                f"omega_d={omega_ds[i]:.6g}, A_charge={A_charges[j]:.6g}, "
                f"lambda={lam:.6g}",
                flush=True,
            )
            save_h5(args.outfile, lambda_matrix, stop_time_matrix, converged_matrix)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(compute_grid_point, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                i, j, lam, stop_time, converged = future.result()
                lambda_matrix[i, j] = lam
                stop_time_matrix[i, j] = stop_time
                converged_matrix[i, j] = converged
                print(
                    f"Finished {completed}/{len(tasks)}: "
                    f"omega_d={omega_ds[i]:.6g}, A_charge={A_charges[j]:.6g}, "
                    f"lambda={lam:.6g}",
                    flush=True,
                )
                save_h5(args.outfile, lambda_matrix, stop_time_matrix, converged_matrix)

    print(f"Saved {args.outfile}", flush=True)


if __name__ == "__main__":
    main()
