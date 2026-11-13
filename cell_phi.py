#!/usr/bin/env python3
# ESPResSo 4.2.2
# Tissue of polymer rings ("cells") + internal active rods (rigid trimers)
#
# Initial prep stage: XY compression or decompression until a target packing
# fraction is reached:
#
#     phi = N * A0 / A
#
# where:
#   N  = number of rings/cells
#   A0 = area of one perfectly circular ring = pi * R^2
#   A  = box area = Lx * Ly
#
# OUTPUT (LAMMPS dump):
#   ITEM: ATOMS id mol cell type x y z

"""Simulate deformable rings containing active rods with ESPResSo 4.2.2.

Run one condition per task:
    pypresso cell_phi.py --paper-index 1 --trajectory-root trajectories

Indices 1 through 144 sweep propulsion (5, 10, 15), membrane epsilon
(0, 1, 2), then nominal packing (0.50 to 1.25 in steps of 0.05).
Each run uses 128 cells and writes 100 frames at intervals of 100000
integration steps. The preset uses the reference input's preparation
and force parameters, with the box rescaling factors recovered from
the campaign boxes. Explicit arguments can override preset values.

Use --dry-run with a paper index to print the full configuration in
ordinary Python. For a job array, pass its one-based task ID as the index.
Warm-up, relaxation and equilibration use the reference input defaults.
The preset records its new initialization seed and can regenerate
independent trajectories with the same model and sampling protocol.
The exact particle paths depend on the original random configuration
and the simulation environment.

ESPResSo setup
https://espressomd.github.io/doc4.2.2/installation.html
"""
from __future__ import annotations

import argparse
import json
import math
import random
from os import link
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    import espressomd


# -----------------------------
# Utilities
# -----------------------------

def min_image(dx: float, L: float) -> float:
    return dx - L * round(dx / L)

def unit(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n < 1e-14:
        return 1.0, 0.0
    return vx / n, vy / n

def folded_pos(pos, box_l, periodicity):
    x, y, z = pos
    if periodicity[0]:
        L = box_l[0]
        x = x - math.floor(x / L) * L
    if periodicity[1]:
        L = box_l[1]
        y = y - math.floor(y / L) * L
    return (x, y, z)

def ramp_value(step: int, ramp_steps: int, f_target: float) -> float:
    if ramp_steps <= 0:
        return f_target
    if step <= 0:
        return 0.0
    if step >= ramp_steps:
        return f_target
    return f_target * (step / float(ramp_steps))

def ring_area0(radius: float) -> float:
    """
    Area occupied by one perfectly circular ring/cell.
    """
    return math.pi * radius * radius

def packing_fraction_phi(n_rings: int, area0: float, box_l) -> float:
    """
    phi = N * A0 / A, with A = Lx * Ly
    """
    A = float(box_l[0]) * float(box_l[1])
    return n_rings * area0 / A

def write_lammpstrj_folded(filename: Path, system: espressomd.System, step: int, pid_to_cell: Dict[int, int]):
    parts = system.part.all()
    box_l = system.box_l
    per = system.periodicity

    xlo, xhi = 0.0, float(box_l[0])
    ylo, yhi = 0.0, float(box_l[1])
    zlo, zhi = 0.0, float(box_l[2])

    bx = "pp" if per[0] else "ff"
    by = "pp" if per[1] else "ff"
    bz = "pp" if per[2] else "ff"

    with filename.open("a", encoding="utf-8") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{int(step)}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(parts)}\n")
        f.write(f"ITEM: BOX BOUNDS {bx} {by} {bz}\n")
        f.write(f"{xlo:.8f} {xhi:.8f}\n")
        f.write(f"{ylo:.8f} {yhi:.8f}\n")
        f.write(f"{zlo:.8f} {zhi:.8f}\n")
        f.write("ITEM: ATOMS id mol cell type x y z\n")
        for p in parts:
            x, y, z = folded_pos(p.pos, box_l, per)
            mol = int(getattr(p, "mol_id", 0))
            cell = int(pid_to_cell.get(int(p.id), 0))
            f.write(f"{int(p.id)} {mol} {cell} {int(p.type)} {x:.8f} {y:.8f} {z:.8f}\n")

def hex_centers(n_rings: int, spacing: float):
    n_cols = int(math.ceil(math.sqrt(n_rings)))
    n_rows = int(math.ceil(n_rings / n_cols))
    dy = spacing * math.sqrt(3.0) / 2.0

    centers = []
    k = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if k >= n_rings:
                break
            x = (c + 0.5 * (r % 2)) * spacing
            y = r * dy
            centers.append((x, y))
            k += 1

    Lx = (n_cols + 1.0) * spacing
    Ly = (n_rows + 1.0) * dy

    shiftx = 0.5 * spacing
    shifty = 0.5 * dy
    centers = [(cx + shiftx, cy + shifty) for (cx, cy) in centers]
    return centers, Lx, Ly


# -----------------------------
# Bonds helper
# -----------------------------

def make_fene_bond(k: float, r0: float, dr_max: float):
    from espressomd.interactions import FeneBond

    try:
        return FeneBond(k=k, r_0=r0, d_r_max=dr_max)
    except TypeError:
        return FeneBond(k=k, r0=r0, dr_max=dr_max)


# -----------------------------
# Langevin per-particle gamma
# -----------------------------

def set_particle_gamma(p, gamma_vec3):
    try:
        p.gamma = list(gamma_vec3)
    except Exception:
        p.gamma = float(gamma_vec3[0])


# -----------------------------
# Box rescaling (XY)
# -----------------------------

def rescale_xy(system: espressomd.System, factor: float):
    """
    Rescale Lx and Ly by 'factor' while rescaling particle positions.
    factor < 1  -> compress
    factor > 1  -> decompress
    """
    if factor <= 0.0:
        raise ValueError("scale factor must be > 0.")
    if abs(factor - 1.0) < 1e-14:
        return

    new_Lx = float(system.box_l[0]) * factor
    new_Ly = float(system.box_l[1]) * factor
    system.change_volume_and_rescale_particles(new_Lx, dir="x")
    system.change_volume_and_rescale_particles(new_Ly, dir="y")


# -----------------------------
# Activity
# -----------------------------

def apply_rods_activity(
    system: espressomd.System,
    rod_triplets: List[Tuple[int, int, int]],  # (end0, mid, end2)
    f_prop: float,
    f_dipole: float,
    dipole_mode: str,
):
    Lx, Ly = float(system.box_l[0]), float(system.box_l[1])
    sgn = -1.0 if dipole_mode == "extensile" else +1.0

    for (e0, mid, e2) in rod_triplets:
        p0 = system.part.by_id(e0)
        p1 = system.part.by_id(mid)
        p2 = system.part.by_id(e2)

        dx = p2.pos[0] - p0.pos[0]
        dy = p2.pos[1] - p0.pos[1]
        if system.periodicity[0]:
            dx = min_image(dx, Lx)
        if system.periodicity[1]:
            dy = min_image(dy, Ly)

        ux, uy = unit(dx, dy)

        p0.ext_force = [0.0, 0.0, 0.0]
        p1.ext_force = [0.0, 0.0, 0.0]
        p2.ext_force = [0.0, 0.0, 0.0]

        if f_prop != 0.0:
            fx = float(f_prop) * ux / 3.0
            fy = float(f_prop) * uy / 3.0
            p0.ext_force = [p0.ext_force[0] + fx, p0.ext_force[1] + fy, 0.0]
            p1.ext_force = [p1.ext_force[0] + fx, p1.ext_force[1] + fy, 0.0]
            p2.ext_force = [p2.ext_force[0] + fx, p2.ext_force[1] + fy, 0.0]

        if f_dipole != 0.0:
            fx0 = sgn * float(f_dipole) * ux
            fy0 = sgn * float(f_dipole) * uy
            fx2 = -fx0
            fy2 = -fy0
            p0.ext_force = [p0.ext_force[0] + fx0, p0.ext_force[1] + fy0, 0.0]
            p2.ext_force = [p2.ext_force[0] + fx2, p2.ext_force[1] + fy2, 0.0]


# -----------------------------
# Main
# -----------------------------

def main():
    campaign = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    campaign.add_argument("--paper-index", type=int, metavar="1..144",
                          help="One-based index of a condition in the article grid.")
    campaign.add_argument("--trajectory-root", type=Path, default=Path("trajectories"),
                          help="Parent directory for article trajectories.")
    preset, _ = campaign.parse_known_args()
    ap = argparse.ArgumentParser(description=__doc__, parents=[campaign],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the resolved input without starting ESPResSo.")

    # Geometry / rings
    ap.add_argument("--n_rings", type=int, default=64)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--lattice_spacing_factor", type=float, default=2.05)

    # Ring mechanics
    ap.add_argument("--k_ring_fene", type=float, default=500.0)
    ap.add_argument("--ring_fene_drmax", type=float, default=1.8)
    ap.add_argument("--k_bend", type=float, default=10.0)
    ap.add_argument("--phi0", type=float, default=math.pi)

    # Rod mechanics
    ap.add_argument("--rho_s", type=float, default=0.05)
    ap.add_argument("--inner_margin", type=float, default=0.85)
    ap.add_argument("--rod_length", type=float, default=2.0)
    ap.add_argument("--rod_bond_r0", type=float, default=1.0)
    ap.add_argument("--k_rod", type=float, default=3000.0)
    ap.add_argument("--k_rod_bend", type=float, default=300.0)
    ap.add_argument("--rod_phi0", type=float, default=math.pi)

    # Activity
    ap.add_argument("--f_prop", type=float, default=10.0)
    ap.add_argument("--f_dipole", type=float, default=0.0)
    ap.add_argument("--dipole_mode", choices=["extensile", "contractile"], default="extensile")
    ap.add_argument("--active_update", type=int, default=1)
    ap.add_argument("--ramp_steps", type=int, default=20000)

    # Thermostat / friction
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345, help="Langevin thermostat seed.")
    ap.add_argument("--init-seed", type=int, help="Seed for initial ring and rod placement.")
    ap.add_argument("--gamma_ring", type=float, default=1.0)
    ap.add_argument("--gamma_rod", type=float, default=1.0)
    ap.add_argument("--gamma_z", type=float, default=None)

    # Integrator
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--skin", type=float, default=0.4)

    # Nonbonded
    ap.add_argument("--epsilon_rep", type=float, default=1.0)
    ap.add_argument("--epsilon_mm", type=float, default=0.0)
    ap.add_argument("--cut_mm", type=float, default=2.5)
    ap.add_argument("--sigma_ringrod", type=float, default=1.25)
    ap.add_argument("--sigma_rodrod", type=float, default=1.0)

    # Init robustness
    ap.add_argument("--overlap_r2", type=float, default=0.5)
    ap.add_argument("--soft_warmup", action="store_true")
    ap.add_argument("--soft_factor", type=float, default=0.1)
    ap.add_argument("--forcecap_warmup", type=float, default=200.0)
    ap.add_argument("--warmup_steps", type=int, default=20000)

    # Prep protocol: target packing fraction
    ap.add_argument("--phi_target", type=float, default=None,
                    help="Target packing fraction phi = N*A0/(Lx*Ly). If None, disables prep rescaling.")
    ap.add_argument("--rescale_factor", type=float, default=0.99,
                    help="XY scale factor applied each prep cycle. <1 compresses, >1 decompresses.")
    ap.add_argument("--prep_max_cycles", type=int, default=1000,
                    help="Maximum number of prep rescaling cycles as a safety stop.")
    ap.add_argument("--phi_tol", type=float, default=1e-4,
                    help="Tolerance for reaching phi_target.")

    ap.add_argument("--compress_relax", type=int, default=5000,
                    help="Relaxation steps after each compression/decompression cycle.")
    ap.add_argument("--compress_forcecap", type=float, default=200.0,
                    help="Force cap during prep relax to avoid blow-ups.")
    ap.add_argument("--compress_disable_activity", action="store_true",
                    help="Disable activity during prep rescaling protocol.")

    # Runs
    ap.add_argument("--equil_steps", type=int, default=500000)
    ap.add_argument("--steps", type=int, default=1100000)
    ap.add_argument("--traj_every", type=int, default=5000)
    ap.add_argument("--traj", type=Path, help="Output trajectory path.")

    if preset.paper_index is not None:
        if not 1 <= preset.paper_index <= 144:
            ap.error("--paper-index must be between 1 and 144")
        force_index, remaining = divmod(preset.paper_index - 1, 48)
        epsilon, density_index = divmod(remaining, 16)
        ap.set_defaults(n_rings=128, f_prop=(5, 10, 15)[force_index],
                        epsilon_mm=epsilon, phi_target=(50 + 5 * density_index) / 100,
                        rescale_factor=None, steps=10_000_000, traj_every=100_000)

    args = ap.parse_args()
    if args.active_update < 1 or not 0 < args.traj_every <= args.steps:
        ap.error("Require --active_update >= 1 and 0 < --traj_every <= --steps.")

    # Centers
    spacing = args.lattice_spacing_factor * args.radius
    centers, Lx, Ly = hex_centers(args.n_rings, spacing)
    A0 = ring_area0(args.radius)
    phi_initial = packing_fraction_phi(args.n_rings, A0, [Lx, Ly])
    if args.rescale_factor is None:
        args.rescale_factor = 1.01 if args.phi_target < phi_initial else 0.995

    if args.traj is None:
        if args.paper_index is None:
            args.traj = Path("traj.lammpstrj")
        else:
            args.traj = (args.trajectory_root / f"fprop{args.f_prop:g}" /
                         f"eps{args.epsilon_mm:g}" / f"traj-phi{args.phi_target:g}.lammpstrj")
    traj_path = args.traj.expanduser()
    if args.init_seed is None:
        args.init_seed = args.seed + (args.paper_index or 0)
    rng = random.Random(args.init_seed)
    configuration = json.dumps(vars(args), indent=2, default=str)
    if args.dry_run:
        print(configuration)
        return
    partial_path = traj_path.with_name(traj_path.name + ".partial")
    if traj_path.exists():
        ap.error(f"Trajectory already exists: {traj_path}")
    if partial_path.exists():
        ap.error(f"Partial trajectory already exists: {partial_path}")

    try:
        import espressomd
        import espressomd.interactions
    except ModuleNotFoundError as exc:
        if exc.name != "espressomd":
            raise
        ap.error("Run this input with ESPResSo 4.2.2 (for example, via pypresso).")
    espressomd.assert_features([
        "LENNARD_JONES", "EXTERNAL_FORCES", "PARTICLE_ANISOTROPY", "THERMOSTAT_PER_PARTICLE"
    ])
    print(configuration, flush=True)

    # System
    system = espressomd.System(box_l=[Lx, Ly, 5.0])
    system.time_step = args.dt
    system.cell_system.skin = args.skin
    system.periodicity = [True, True, False]
    system.integrator.set_vv()

    type_ring = 0
    type_rod = 1

    system.thermostat.turn_off()
    system.thermostat.set_langevin(
        kT=args.temperature,
        gamma=[1.0, 1.0, 1.0],
        seed=args.seed,
    )

    # Bonds
    ring_bond = make_fene_bond(k=args.k_ring_fene, r0=args.r0, dr_max=args.ring_fene_drmax)
    system.bonded_inter.add(ring_bond)
    ring_bend = espressomd.interactions.AngleHarmonic(bend=args.k_bend, phi0=args.phi0)
    system.bonded_inter.add(ring_bend)

    rod_bond = espressomd.interactions.HarmonicBond(k=args.k_rod, r_0=args.rod_bond_r0)
    system.bonded_inter.add(rod_bond)
    rod_angle = espressomd.interactions.AngleHarmonic(bend=args.k_rod_bend, phi0=args.rod_phi0)
    system.bonded_inter.add(rod_angle)

    def wca_params(eps: float, sig: float):
        return dict(epsilon=eps, sigma=sig, cutoff=(2.0 ** (1.0 / 6.0)) * sig, shift="auto")

    # Membrane-membrane
    if args.epsilon_mm > 0.0:
        system.non_bonded_inter[type_ring, type_ring].lennard_jones.set_params(
            epsilon=args.epsilon_mm, sigma=args.sigma, cutoff=args.cut_mm * args.sigma, shift=0.0
        )
        mm_mode = "LJ"
    else:
        system.non_bonded_inter[type_ring, type_ring].lennard_jones.set_params(
            **wca_params(args.epsilon_rep, args.sigma)
        )
        mm_mode = "WCA"

    system.non_bonded_inter[type_ring, type_rod].lennard_jones.set_params(
        **wca_params(args.epsilon_rep, args.sigma_ringrod * args.sigma)
    )
    system.non_bonded_inter[type_rod, type_rod].lennard_jones.set_params(
        **wca_params(args.epsilon_rep, args.sigma_rodrod * args.sigma)
    )

    n_beads = max(6, int(2.0 * math.pi * args.radius / args.r0))
    a_ring = math.pi * max(0.0, (args.radius - 0.5 * args.sigma)) ** 2
    n_rods_per_ring = max(0, int(a_ring * args.rho_s))

    gz_ring = args.gamma_ring if args.gamma_z is None else args.gamma_z
    gz_rod = args.gamma_rod if args.gamma_z is None else args.gamma_z
    gamma_ring_vec = [args.gamma_ring, args.gamma_ring, gz_ring]
    gamma_rod_vec = [args.gamma_rod, args.gamma_rod, gz_rod]

    mol_counter = 1
    rod_triplets: List[Tuple[int, int, int]] = []
    pid_to_cell: Dict[int, int] = {}

    # Build rings + rods
    for (cx, cy) in centers:
        ring_mol = mol_counter
        mol_counter += 1

        theta0 = 2.0 * math.pi * rng.random()
        ring_ids: List[int] = []

        for i in range(n_beads):
            theta = theta0 + 2.0 * i * math.pi / n_beads
            p = system.part.add(
                pos=[cx + args.radius * math.cos(theta), cy + args.radius * math.sin(theta), 0.0],
                type=type_ring, mol_id=ring_mol
            )
            p.fix = [False, False, True]
            pid_to_cell[p.id] = ring_mol
            set_particle_gamma(p, gamma_ring_vec)

            ring_ids.append(p.id)
            if i > 0:
                system.part.by_id(ring_ids[i - 1]).add_bond((ring_bond, p))

        system.part.by_id(ring_ids[-1]).add_bond((ring_bond, system.part.by_id(ring_ids[0])))

        for i in range(n_beads):
            im1 = ring_ids[(i - 1) % n_beads]
            i0 = ring_ids[i]
            ip1 = ring_ids[(i + 1) % n_beads]
            system.part.by_id(i0).add_bond((ring_bend, im1, ip1))

        # rods inside
        r_in = max(0.0, args.radius - args.inner_margin)
        r_in2 = r_in * r_in
        half_len = 0.5 * args.rod_length

        for _ in range(n_rods_per_ring):
            placed = False
            trials = 0
            while (not placed) and (trials < 20000):
                trials += 1

                x = cx + r_in * (2.0 * rng.random() - 1.0)
                y = cy + r_in * (2.0 * rng.random() - 1.0)
                if (x - cx) ** 2 + (y - cy) ** 2 > r_in2:
                    continue

                ang = 2.0 * math.pi * rng.random()
                ux = math.cos(ang)
                uy = math.sin(ang)

                x0, y0 = x - half_len * ux, y - half_len * uy
                x1, y1 = x, y
                x2, y2 = x + half_len * ux, y + half_len * uy

                if ((x0 - cx) ** 2 + (y0 - cy) ** 2 > r_in2) or ((x2 - cx) ** 2 + (y2 - cy) ** 2 > r_in2):
                    continue

                overlap = False
                for pj in system.part.all():
                    dx = x0 - pj.pos[0]; dy = y0 - pj.pos[1]
                    if dx * dx + dy * dy < args.overlap_r2:
                        overlap = True
                        break
                    dx = x1 - pj.pos[0]; dy = y1 - pj.pos[1]
                    if dx * dx + dy * dy < args.overlap_r2:
                        overlap = True
                        break
                    dx = x2 - pj.pos[0]; dy = y2 - pj.pos[1]
                    if dx * dx + dy * dy < args.overlap_r2:
                        overlap = True
                        break
                if overlap:
                    continue

                rod_mol = mol_counter
                mol_counter += 1

                p0 = system.part.add(pos=[x0, y0, 0.0], type=type_rod, mol_id=rod_mol, fix=[False, False, True])
                p1 = system.part.add(pos=[x1, y1, 0.0], type=type_rod, mol_id=rod_mol, fix=[False, False, True])
                p2 = system.part.add(pos=[x2, y2, 0.0], type=type_rod, mol_id=rod_mol, fix=[False, False, True])

                pid_to_cell[p0.id] = ring_mol
                pid_to_cell[p1.id] = ring_mol
                pid_to_cell[p2.id] = ring_mol

                set_particle_gamma(p0, gamma_rod_vec)
                set_particle_gamma(p1, gamma_rod_vec)
                set_particle_gamma(p2, gamma_rod_vec)

                system.part.by_id(p0.id).add_bond((rod_bond, p1))
                system.part.by_id(p1.id).add_bond((rod_bond, p2))
                system.part.by_id(p1.id).add_bond((rod_angle, p0.id, p2.id))

                rod_triplets.append((p0.id, p1.id, p2.id))
                placed = True

            if not placed:
                if args.paper_index is not None:
                    raise RuntimeError(f"Could not place all rods in cell {ring_mol}.")
                print("[WARN] Could not place a rod in a ring. Reduce rho_s or increase inner_margin/overlap_r2.")

    phi_init = packing_fraction_phi(args.n_rings, A0, system.box_l)

    print(f"[INFO] Built: n_rings={args.n_rings}, beads/ring={n_beads}, rods/ring={n_rods_per_ring}, total_rods={len(rod_triplets)}")
    print(f"[INFO] Membrane-membrane mode: {mm_mode} (epsilon_mm={args.epsilon_mm})")
    print(f"[INFO] dt={args.dt}  f_prop={args.f_prop}  f_dipole={args.f_dipole}")
    print(f"[INFO] A0={A0:.6f}  phi_init={phi_init:.6f}")

    traj_path.parent.mkdir(parents=True, exist_ok=True)
    # A partial trajectory is excluded from the analysis file search.
    with partial_path.open("x", encoding="utf-8"):
        pass

    # Soft warmup (optional)
    def soften(eps_val: float):
        if args.epsilon_mm <= 0.0:
            system.non_bonded_inter[type_ring, type_ring].lennard_jones.set_params(**wca_params(eps_val, args.sigma))
        system.non_bonded_inter[type_ring, type_rod].lennard_jones.set_params(**wca_params(eps_val, args.sigma_ringrod * args.sigma))
        system.non_bonded_inter[type_rod, type_rod].lennard_jones.set_params(**wca_params(eps_val, args.sigma_rodrod * args.sigma))

    if args.soft_warmup:
        soften(args.soft_factor * args.epsilon_rep)

    # Warmup: no activity
    for (e0, mid, e2) in rod_triplets:
        system.part.by_id(e0).ext_force = [0.0, 0.0, 0.0]
        system.part.by_id(mid).ext_force = [0.0, 0.0, 0.0]
        system.part.by_id(e2).ext_force = [0.0, 0.0, 0.0]

    system.force_cap = args.forcecap_warmup
    system.integrator.run(args.warmup_steps)
    system.force_cap = 0.0

    if args.soft_warmup:
        soften(args.epsilon_rep)

    # -------------------------
    # Initial XY rescaling until target phi
    # -------------------------
    if args.phi_target is not None:
        phi_now = packing_fraction_phi(args.n_rings, A0, system.box_l)

        if args.phi_target <= 0.0:
            raise ValueError("--phi_target must be > 0.")
        if abs(args.rescale_factor - 1.0) < 1e-14:
            raise ValueError("--rescale_factor cannot be 1.0 when using --phi_target.")

        print(f"[PREP] Target phi={args.phi_target:.6f} | current phi={phi_now:.6f}")

        if phi_now < args.phi_target and args.rescale_factor > 1.0:
            raise ValueError("Current phi < phi_target, so compression is needed: use --rescale_factor < 1.")
        if phi_now > args.phi_target and args.rescale_factor < 1.0:
            raise ValueError("Current phi > phi_target, so decompression is needed: use --rescale_factor > 1.")

        prev_force_cap = getattr(system, "force_cap", 0.0)
        system.force_cap = args.compress_forcecap

        cycle = 0
        while cycle < args.prep_max_cycles:
            phi_now = packing_fraction_phi(args.n_rings, A0, system.box_l)
            err = abs(phi_now - args.phi_target)

            if err <= args.phi_tol:
                break

            rescale_xy(system, args.rescale_factor)

            # relax after each rescaling
            steps_left = args.compress_relax
            local_step = 0
            while steps_left > 0:
                block = min(args.active_update, steps_left)

                if args.compress_disable_activity:
                    for (e0, mid, e2) in rod_triplets:
                        system.part.by_id(e0).ext_force = [0.0, 0.0, 0.0]
                        system.part.by_id(mid).ext_force = [0.0, 0.0, 0.0]
                        system.part.by_id(e2).ext_force = [0.0, 0.0, 0.0]
                else:
                    fprop = ramp_value(local_step, args.ramp_steps, args.f_prop)
                    fdip = ramp_value(local_step, args.ramp_steps, args.f_dipole)
                    apply_rods_activity(
                        system,
                        rod_triplets,
                        f_prop=fprop,
                        f_dipole=fdip,
                        dipole_mode=args.dipole_mode
                    )

                system.integrator.run(block)
                steps_left -= block
                local_step += block

            cycle += 1
            phi_now = packing_fraction_phi(args.n_rings, A0, system.box_l)
            print(
                f"[PREP] cycle {cycle:4d} | "
                f"box=({system.box_l[0]:.6f}, {system.box_l[1]:.6f}) | "
                f"phi={phi_now:.6f}"
            )

            # break if target is reached or slightly crossed within tolerance
            if args.rescale_factor < 1.0 and phi_now >= args.phi_target - args.phi_tol:
                break
            if args.rescale_factor > 1.0 and phi_now <= args.phi_target + args.phi_tol:
                break

        system.force_cap = prev_force_cap

        phi_now = packing_fraction_phi(args.n_rings, A0, system.box_l)
        print(f"[PREP] Finished prep at phi={phi_now:.6f} after {cycle} cycles.")

    # Equilibration (ramped activity)
    step = 0
    while step < args.equil_steps:
        block = min(args.active_update, args.equil_steps - step)
        fprop = ramp_value(step, args.ramp_steps, args.f_prop)
        fdip = ramp_value(step, args.ramp_steps, args.f_dipole)
        apply_rods_activity(system, rod_triplets, f_prop=fprop, f_dipole=fdip, dipole_mode=args.dipole_mode)
        system.integrator.run(block)
        step += block

    print("[OK] Finished equilibration with ramp.")

    # Production
    step = 0
    next_dump = args.traj_every
    while step < args.steps:
        remaining = args.steps - step
        block = min(args.active_update, remaining, next_dump - step if next_dump > step else remaining)

        fprop = ramp_value(step, args.ramp_steps, args.f_prop)
        fdip = ramp_value(step, args.ramp_steps, args.f_dipole)
        apply_rods_activity(system, rod_triplets, f_prop=fprop, f_dipole=fdip, dipole_mode=args.dipole_mode)

        system.integrator.run(block)
        step += block

        if step >= next_dump:
            write_lammpstrj_folded(partial_path, system, step=step, pid_to_cell=pid_to_cell)
            next_dump += args.traj_every

    link(partial_path, traj_path)
    partial_path.unlink()
    print(f"[OK] Finished. Trajectory written to: {traj_path.resolve()}")


if __name__ == "__main__":
    main()
