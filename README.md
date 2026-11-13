# Crowding increases internal rod order and slows deformable active rings

This repository contains the simulation and analysis scripts for a two-dimensional
model of deformable rings driven by self-propelled rods. The workflow measures
ring shape, internal rod order, and cell translation, and compares speeds at
matched contour area fraction.

## Run the simulations

Use **ESPResSo 4.2.2** with `LENNARD_JONES`, `EXTERNAL_FORCES`,
`PARTICLE_ANISOTROPY`, and `THERMOSTAT_PER_PARTICLE` enabled. The examples
use the `pypresso` launcher from that installation.
[ESPResSo installation guide](https://espressomd.github.io/doc4.2.2/installation.html)

The paper preset in [`cell_phi.py`](cell_phi.py) runs one trajectory for each
combination in the following grid.

| Parameter | Values |
| --- | --- |
| Propulsion force per rod | 5, 10, 15 |
| Membrane interaction parameter | 0, 1, 2 |
| Target nominal packing fraction | 0.50 to 1.25 in increments of 0.05 |

Each run contains 128 rings with 14 rods per ring. Production lasts
10 million integration steps with a step size of 0.001. The script saves
100 frames, one every 100,000 steps.

Inspect the first configuration with standard Python, then run it with ESPResSo.

```bash
python3 cell_phi.py --paper-index 1 --dry-run
pypresso cell_phi.py --paper-index 1 --trajectory-root trajectories
```

Use indices 1 through 144 to generate the full campaign. Packing fraction
varies fastest, followed by the membrane interaction parameter and then the
propulsion force. For a job array, pass the one-based task index to
`--paper-index`. The example above writes
`trajectories/fprop5/eps0/traj-phi0.5.lammpstrj`.

The preset uses the preparation settings from the reference input and
reports the full configuration, including the thermostat and initialization
seeds. The original particle dumps are kept outside the repository because of
their size. Reruns generate new trajectories at the same state points.

## Analyze trajectories and generate figures

The analysis was verified with Python 3.12 and the package versions below.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5 pandas==2.2.3 matplotlib==3.10.8
```

Run the following commands from the repository root. The first command also
accepts existing `.lammpstrj` or `.lammpstrj.gz` files. Keep the force and
membrane interaction parameter in the directory names and the nominal packing
fraction in the filename, as in the example above.

```bash
python analyze_cells_rods.py --root trajectories --out cell_analysis \
    --dt 0.001 --radius0 10 --no-plots --stop-on-error

python compare_cells_rods.py --root cell_analysis --out comparisons \
    --speed-lag 100 --blocks 6 --no-plots

python analisar_resumos_celulas.py comparisons --out results --no-plots

python make_figures.py --tables results/tabelas --out figures
```

The speed and directional displacement use pairs of consecutive saved frames,
separated by 100 simulation time units. The analysis takes the largest
standard error across the five temporal block partitions. The matched-area
comparisons use linear interpolation within the overlapping range of the
measured contour area fractions.

| Output | Contents |
| --- | --- |
| `cell_analysis/runs/` | Ring geometry, rod orientation, and center-of-mass positions for each cell and frame |
| `comparisons/` | Condition averages, block statistics, polarization autocorrelations, and speed contrasts |
| `results/tabelas/` | Manuscript tables, including the speed contrasts at contour area fraction 0.55 in `contrastes_area_comum.csv` |
| `figures/` | `model.pdf`, `structure.pdf`, `transport.pdf`, `attraction.pdf`, and the appendix figure `polarization.pdf` |

The manuscript figure script expects one trajectory at each of the 144 state
points. Use a fresh analysis directory when processing another campaign.

## Citation

If you use these scripts, please cite *Crowding increases internal rod order
and slows deformable active rings*.
