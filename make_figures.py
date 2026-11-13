"""Reproduce the figures for Crowding increases internal rod order and slows deformable active rings.

Install NumPy, pandas and Matplotlib in the analysis Python environment.
The figures were checked with NumPy 2.3.5, pandas 2.2.3 and Matplotlib 3.10.8.
After generating the trajectories with cell_phi.py, run from the script folder:

    python analyze_cells_rods.py --root trajectories --out cell_analysis --dt 0.001 --radius0 10 --no-plots
    python compare_cells_rods.py --root cell_analysis --out comparisons --speed-lag 100 --blocks 6 --no-plots
    python analisar_resumos_celulas.py comparisons --out results --no-plots
    python make_figures.py --tables results/tabelas --out figures

The trajectory reader accepts .lammpstrj and .lammpstrj.gz files with the
id, mol, cell, type, x, y and z fields written by the simulation input.
Use a separate output directory when analyzing another set of trajectories.
The PDFs are model, structure, transport, attraction and polarization,
in manuscript order. The tables are written by analisar_resumos_celulas.py.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


COLORS = {0: "#225F97", 1: "#AA541B", 2: "#157C69"}
MARKERS = {0: "o", 1: "s", 2: "^"}
FORCES = (5, 10, 15)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def panel(ax, letter):
    ax.text(0, 1.035, f"({letter})", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom")


def legend_epsilon(values=(0, 1, 2)):
    return [
        Line2D([], [], lw=1.15, marker=MARKERS[e], color=COLORS[e],
               markersize=3.6, label=rf"$\epsilon={e}$")
        for e in values
    ]


def plot_quantity(ax, rows, x, y, *, style="-", scale=1):
    for eps, group in rows.groupby("epsilon"):
        g = group.sort_values(x)
        ax.errorbar(g[x], scale * g[y], yerr=scale * g[y + "_ep_max"],
                    color=COLORS[eps], linestyle=style, marker=MARKERS[eps],
                    lw=1.05, markersize=2.5, elinewidth=.60, capsize=1.25)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def nominal_axis(ax):
    ax.set_xlabel(r"Nominal packing $\phi$")
    ax.set_xticks([.50, .75, 1.00, 1.25], ["0.50", "0.75", "1.00", "1.25"])


def schematic(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.65, 2.42),
                               gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.035, right=.98, bottom=.22, top=.86, wspace=.32)
    a.set(aspect="equal", xlim=(-11.1, 13.4), ylim=(-8.8, 8.8))
    a.axis("off")
    angle = np.linspace(0, 2 * np.pi, 62, endpoint=False)
    # A schematic closed membrane, with an interior set of 14 trimers.
    x = 8.2 * np.cos(angle) + .40 * np.cos(2 * angle)
    y = 6.9 * np.sin(angle) + .24 * np.sin(3 * angle)
    a.fill(x, y, color="#EFF3F7")
    a.plot(np.r_[x, x[0]], np.r_[y, y[0]], color="#536779", lw=.7)
    a.scatter(x, y, color="#536779", s=5.0, zorder=3)
    rods = [
        (-3.9, -3.7, 30), (-.6, -4.0, 34), (2.8, -3.6, 46), (5.0, -1.9, 96),
        (-5.0, -.9, 65), (-2.2, -1.3, 1), (2.7, -1.1, -14), (5.8, .9, 30),
        (-4.9, 2.3, 39), (-2.1, 1.5, -38), (1.7, 1.9, 64), (4.6, 3.6, 141),
        (-2.2, 4.3, 12), (1.0, 4.8, 184),
    ]
    direction_sum = np.zeros(2)
    for cx, cy, degrees in rods:
        u = np.array([np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))])
        direction_sum += u
        center = np.array([cx, cy])
        for offset in (-.56, 0, .56):
            a.add_patch(Circle(center + offset * u, .33, fc="#D18848",
                               ec="#8F491E", lw=.48, zorder=4))
        a.add_patch(FancyArrowPatch(center + .86 * u, center + 1.52 * u,
                                    arrowstyle="-|>", mutation_scale=6,
                                    color="#8F491E", lw=.60, zorder=5))
    a.plot(0, 0, "+", color="#134F47", ms=5, zorder=6)
    endpoint = 3.85 * direction_sum / np.linalg.norm(direction_sum)
    a.add_patch(FancyArrowPatch((0, 0), endpoint, arrowstyle="-|>",
                                mutation_scale=12, lw=1.35, color="#134F47", zorder=6))
    a.text(.07, -.48, r"$\mathbf{R}_i$", color="#134F47", ha="right", va="top")
    a.annotate(r"$\mathbf{F}_i^{\rm a}$", xy=endpoint * .86, xytext=(8.35, 4.75),
               ha="left", va="center", color="#134F47",
               arrowprops={"arrowstyle": "-", "color": "#134F47", "lw": .7})
    a.annotate("membrane", xy=(-6.2, 4.05), xytext=(-10.8, 7.8), ha="left",
               color="#374E61", arrowprops={"arrowstyle": "-", "color": "#536779",
               "lw": .7})
    a.annotate("active trimer", xy=(5.0, -2.12), xytext=(8.7, -5.0),
               color="#8F491E", ha="left", va="center",
               arrowprops={"arrowstyle": "-", "color": "#8F491E", "lw": .7})
    panel(a, "a")

    r = np.linspace(1.067, 2.70, 1000)
    r_min = 2 ** (1 / 6)
    lj_force = 24 * (2 * r**-13 - r**-7)
    for eps, cutoff, style in ((2, 2.5, "-"), (1, 2.5, "-"), (0, r_min, "--")):
        b.plot(r, np.where(r <= cutoff, max(1, eps) * lj_force, 0.),
               color=COLORS[eps], ls=style, lw=1.25, label=rf"$\epsilon={eps}$")
    b.axhline(0, lw=.6, color=".55", zorder=0)
    b.axvline(r_min, lw=.6, ls=":", color=".50")
    b.set(xlabel=r"Bead separation $r/\sigma$", ylabel=r"Membrane pair force $F_{\rm mm}(r)$",
          xlim=(1.068, 2.65), ylim=(-5.55, 10.35), xticks=[1.1, 1.5, 2, 2.5])
    b.legend(frameon=False, loc="upper right", fontsize=8, handlelength=1.9)
    panel(b, "b")
    save(fig, out, "model")


def structure(data, out):
    fig, axes = plt.subplots(3, 3, sharex=True, sharey="row", figsize=(6.65, 4.78))
    fig.subplots_adjust(left=.105, right=.99, top=.845, bottom=.105, wspace=.18, hspace=.32)
    styles = [
        Line2D([], [], color=".23", lw=1.1, label=r"$\lambda_{\min},\, P$"),
        Line2D([], [], color=".23", ls="--", lw=1.1, label=r"$\lambda_{\max},\, S$"),
    ]
    fig.legend(handles=legend_epsilon() + styles, ncol=5, frameon=False, fontsize=8.5,
               loc="upper center", bbox_to_anchor=(.515, .995), columnspacing=1.2)
    for j, force in enumerate(FORCES):
        g = data[data["f_prop"] == force]
        a, b, c = axes[:, j]
        a.set_title(rf"$f_{{\rm prop}}={force}$", pad=7)
        plot_quantity(a, g, "phi_nominal", "lambda_min")
        plot_quantity(a, g, "phi_nominal", "lambda_max", style="--")
        plot_quantity(b, g, "phi_nominal", "asphericity")
        plot_quantity(c, g, "phi_nominal", "P")
        plot_quantity(c, g, "phi_nominal", "S", style="--")
        nominal_axis(c)
    for i, label in enumerate((r"Gyration eigenvalues", r"Asphericity $\langle b\rangle$",
                               r"Internal order $P,\,S$")):
        axes[i, 0].set_ylabel(label)
    for i, ax in enumerate(axes.flat):
        panel(ax, chr(97 + i))
    save(fig, out, "structure")


def transport(data, out):
    fig, axes = plt.subplots(3, 3, sharex=True, sharey="row", figsize=(6.65, 4.78))
    fig.subplots_adjust(left=.11, right=.99, top=.845, bottom=.105, wspace=.16, hspace=.32)
    fig.legend(handles=legend_epsilon(), ncol=3, frameon=False, fontsize=8.5,
               loc="upper center", bbox_to_anchor=(.53, .99))
    for j, force in enumerate(FORCES):
        g = data[data["f_prop"] == force]
        a, b, c = axes[:, j]
        a.set_title(rf"$f_{{\rm prop}}={force}$", pad=7)
        plot_quantity(a, g, "phi_nominal", "speed")
        plot_quantity(b, g, "phi_nominal", "speed_per_force", scale=1000.)
        plot_quantity(c, g, "phi_nominal", "v_parallel", scale=g["speed_tau"].iloc[0])
        nominal_axis(c)
    labels = (r"Speed $\bar v$", r"$10^3\,\bar v/\langle F^{\rm a}\rangle$",
              r"$L_\parallel(\Delta t)$")
    for i, label in enumerate(labels):
        axes[i, 0].set_ylabel(label)
    for i, ax in enumerate(axes.flat):
        panel(ax, chr(97 + i))
    save(fig, out, "transport")


def polarization(data, out):
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(6.65, 3.55))
    fig.subplots_adjust(left=.105, right=.99, top=.79, bottom=.14, wspace=.17, hspace=.33)
    fig.legend(handles=legend_epsilon(), ncol=3, frameon=False, fontsize=8.5,
               loc="upper center", bbox_to_anchor=(.53, .985))
    for j, force in enumerate(FORCES):
        g = data[data["f_prop"] == force]
        a, b = axes[:, j]
        a.set_title(rf"$f_{{\rm prop}}={force}$", pad=7)
        plot_quantity(a, g, "phi_nominal", "C_p", style="none")
        plot_quantity(b, g, "phi_nominal", "C_direction", style="none")
        nominal_axis(b)
    for i, ax in enumerate(axes.flat):
        ax.axhline(0, color=".58", lw=.7, ls="--", zorder=0)
        panel(ax, chr(97 + i))
    axes[0, 0].set_ylabel(r"$C_p(\Delta t)$")
    axes[1, 0].set_ylabel(r"$C_{\hat p}(\Delta t)$")
    save(fig, out, "polarization")


def attraction(data, matched, out):
    fig, axes = plt.subplots(2, 3, figsize=(6.65, 3.68), sharey=False)
    fig.subplots_adjust(left=.112, right=.99, top=.78, bottom=.15, wspace=.31, hspace=.53)
    styles = [
        Line2D([], [], color=".23", ls="--", lw=1.1, label=r"Fixed $\phi$"),
        Line2D([], [], color=".23", lw=1.1, label=r"Matched $\phi_A$"),
    ]
    fig.legend(handles=legend_epsilon() + styles, ncol=5, frameon=False, fontsize=8.5,
               loc="upper center", bbox_to_anchor=(.52, .989), columnspacing=1.0)
    for j, force in enumerate(FORCES):
        a, b = axes[:, j]
        g = data[data["f_prop"] == force]
        a.set_title(rf"$f_{{\rm prop}}={force}$", pad=7)
        plot_quantity(a, g, "phi_area", "speed")
        a.axvline(.55, color=".55", lw=.75, ls=":")
        a.set_xlabel(r"Contour area fraction $\phi_A$")
        a.xaxis.set_major_locator(MaxNLocator(4))
        for (eps, kind), curves in matched[matched["f_prop"] == force].groupby(
                ["epsilon", "coverage_type"]):
            g = curves.sort_values("phi_nominal_reference")
            x = g["phi_nominal_reference"].to_numpy()
            b.plot(x, g["delta_speed_percent"], ls="-" if kind == "contours" else "--",
                   color=COLORS[eps], lw=1.05)
            if kind == "contours":
                b.fill_between(x, g["sensitivity_low_pct"].to_numpy(),
                               g["sensitivity_high_pct"].to_numpy(),
                               color=COLORS[eps], alpha=.15, linewidth=0)
        b.axhline(0, color=".58", lw=.70)
        b.set(xlabel=r"Reference $\phi$ at $\epsilon=0$")
        b.set_xticks([.50, .75, 1.00, 1.25], ["0.50", "0.75", "1.00", "1.25"])
        b.set_yticks([-30, -20, -10, 0, 10])
        if j:
            b.tick_params(labelleft=False)
    lower = matched[["delta_speed_percent", "sensitivity_low_pct"]].min().min()
    upper = matched[["delta_speed_percent", "sensitivity_high_pct"]].max().max()
    for ax in axes[1]:
        ax.set_ylim(min(-32, lower - 2), max(10, upper + 2))
    axes[0, 0].set_ylabel(r"Speed $\bar v$")
    axes[1, 0].set_ylabel(r"Change in speed (%)")
    for i, ax in enumerate(axes.flat):
        panel(ax, chr(97 + i))
    save(fig, out, "attraction")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True,
                        help="The tabelas directory produced by analisar_resumos_celulas.py.")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    data = pd.read_csv(args.tables / "condicoes_ep_temporal.csv")
    matched = pd.read_csv(args.tables / "contrastes_em_area_e_phi.csv")
    keys = ["f_prop", "epsilon", "phi_nominal"]
    expected = {(f, eps, p / 100) for f in FORCES for eps in COLORS for p in range(50, 126, 5)}
    if data.duplicated(keys).any() or set(data[keys].itertuples(index=False, name=None)) != expected:
        parser.error("The figures require one run at every state point in the 144-condition grid.")
    if data["speed_tau"].nunique() != 1 or not np.allclose(data["speed_tau"], data["frame_dt"]):
        parser.error("The figures require one common first-frame lag.")
    args.out.mkdir(parents=True, exist_ok=True)
    schematic(args.out)
    structure(data, args.out)
    transport(data, args.out)
    attraction(data, matched, args.out)
    polarization(data, args.out)


if __name__ == "__main__":
    main()
