#!/usr/bin/env python3
"""Analisa os CSVs gerados por compare_cells_rods.py.

Uso: python analisar_resumos_celulas.py comparacoes --out resultados
Também aceita um ZIP com os mesmos CSVs no diretório raiz.
Requer NumPy, pandas e Matplotlib.

EP máximo é o maior erro padrão temporal entre as partições disponíveis,
com pelo menos quatro blocos válidos. As faixas de contraste variam as
entradas em ±2 EP máximos. São faixas de sensibilidade, sem interpretação
de intervalo de confiança. A interpolação é linear e restrita ao suporte
das curvas medidas.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


KEY = ["f_prop", "epsilon", "phi_nominal"]
METRICS = ["phi_area", "area", "Rg2_ring", "asphericity", "lambda_max", "lambda_min",
           "P", "S", "rods_per_area", "rods_shape_cos2", "speed", "speed_per_force",
           "v_parallel", "C_p"]
COLORS = {0.: "#246AA6", 1.: "#B76014", 2.: "#24805C"}
MARKERS = {0.: "o", 1.: "s", 2.: "^"}


def read_comparisons(source):
    names = ["runs_summary.csv", "conditions_summary.csv", "block_statistics.csv",
             "matched_coverage.csv", "orientation_dynamics.csv"]
    if source.is_dir():
        return [pd.read_csv(source / name) for name in names]
    with ZipFile(source) as archive:
        return [pd.read_csv(archive.open(name)) for name in names]


def validate(runs, conditions, blocks, dynamics):
    if runs["run"].duplicated().any() or runs.duplicated(KEY).any():
        raise ValueError("Esta análise requer uma trajetória por condição.")
    if not (conditions["n_trajectories"] == 1).all():
        raise ValueError("A tabela de condições contém réplicas.")
    check = runs.merge(conditions, on=KEY, validate="one_to_one")
    if len(check) != len(runs) or len(check) != len(conditions):
        raise ValueError("Cobertura diferente entre condições e trajetórias.")
    for field in METRICS:
        np.testing.assert_allclose(check[field], check[field + "_mean"], atol=1e-12)
    if set(runs["run"]) != set(blocks["run"]) or set(runs["run"]) != set(dynamics["run"]):
        raise ValueError("Séries de blocos ou dinâmica com cobertura diferente.")
    if (runs["n_complete_area_frames"] != runs["n_frames"]).any():
        raise ValueError("Há frames incompletos no cálculo da área.")
    if runs["speed_tau"].nunique() != 1:
        raise ValueError("A campanha precisa de um atraso comum para a velocidade.")
    if not np.allclose(runs["speed_tau"], runs["frame_dt"]):
        raise ValueError("A projeção desta análise deve usar o primeiro atraso amostrado.")
    np.testing.assert_allclose(runs["Rg2_ring"], runs["lambda_max"] + runs["lambda_min"],
                               atol=1e-10)


@dataclass
class Curve:
    x: np.ndarray
    y: np.ndarray
    dx: np.ndarray
    dy: np.ndarray

    @classmethod
    def from_rows(cls, rows, abscissa):
        rows = rows.sort_values(abscissa)
        x = rows[abscissa].to_numpy()
        if len(x) < 2 or (np.diff(x) <= 0).any():
            raise ValueError(f"A curva de {abscissa} precisa de abscissas distintas.")
        dx = (2 * rows[abscissa + "_ep_max"].to_numpy()
              if abscissa == "phi_area" else np.zeros(len(x)))
        return cls(x, rows["speed"].to_numpy(), dx, 2 * rows["speed_ep_max"].to_numpy())

    def value_and_band(self, point, half_width=0.):
        """Média interpolada e faixa externa para perturbações em x e y."""
        if not self.x[0] <= point <= self.x[-1]:
            return np.nan, np.nan, np.nan
        central = float(np.interp(point, self.x, self.y))
        xmin, xmax = point - half_width, point + half_width
        xlo, xhi = self.x - self.dx, self.x + self.dx
        ylo, yhi = self.y - self.dy, self.y + self.dy
        # A abscissa deve permanecer dentro da curva sob todas as perturbações.
        if xmin < xhi.min() or xmax > xlo.max():
            return central, np.nan, np.nan
        low, high = [], []
        indices = np.arange(len(self.x))
        # A vizinhança pode mudar se as faixas de duas abscissas se cruzarem.
        for j, k in permutations(indices, 2):
            lo_point = max(xmin, xlo[j])
            hi_point = min(xmax, xhi[k])
            if lo_point > hi_point:
                continue
            left_max = min(xhi[j], hi_point)
            right_min = max(xlo[k], lo_point)
            between = (xlo > left_max) & (xhi < right_min)
            if np.any(between & (indices != j) & (indices != k)):
                continue
            if left_max >= right_min:
                wlo, whi = 0., 1.
            else:
                wlo = np.clip((lo_point - left_max) / (xhi[k] - left_max), 0., 1.)
                whi = np.clip((hi_point - xlo[j]) / (right_min - xlo[j]), 0., 1.)
            for w in (wlo, whi):
                low.append((1 - w) * ylo[j] + w * ylo[k])
                high.append((1 - w) * yhi[j] + w * yhi[k])
        return central, float(min(low)), float(max(high))


def contrast(ref, other):
    v0, lo0, hi0 = ref
    v, lo, hi = other
    if v0 <= 0:
        raise ValueError("A velocidade de referência precisa ser positiva.")
    central = 100 * (v / v0 - 1)
    if not np.isfinite([lo0, hi0, lo, hi]).all() or lo0 <= 0:
        return central, np.nan, np.nan
    return central, 100 * (lo / hi0 - 1), 100 * (hi / lo0 - 1)


def matched_rows(data, common_area):
    rows, common = [], []
    for force, family in data.groupby("f_prop"):
        base = family[family["epsilon"] == 0].sort_values("phi_nominal")
        if len(base) < 2 or family["epsilon"].nunique() < 2:
            raise ValueError(f"A força {force:g} requer a referência epsilon=0 e outra curva.")
        for epsilon, selected in family[family["epsilon"] != 0].groupby("epsilon"):
            for abscissa, kind in (("phi_nominal", "nominal"), ("phi_area", "contours")):
                curve = Curve.from_rows(selected, abscissa)
                for _, row in base.iterrows():
                    dx = 2 * row["phi_area_ep_max"] if kind == "contours" else 0.
                    v, vlow, vhigh = curve.value_and_band(row[abscissa], dx)
                    if not np.isfinite(v):
                        continue
                    v0, dv0 = row["speed"], 2 * row["speed_ep_max"]
                    value, low, high = contrast((v0, v0 - dv0, v0 + dv0), (v, vlow, vhigh))
                    rows.append(dict(f_prop=force, epsilon=epsilon, coverage_type=kind,
                        coverage=row[abscissa], phi_nominal_reference=row["phi_nominal"],
                        speed_reference=v0, speed_interpolated=v, delta_speed_percent=value,
                        sensitivity_low_pct=low, sensitivity_high_pct=high))
                if kind == "contours":
                    ref = Curve.from_rows(base, abscissa).value_and_band(common_area)
                    other = curve.value_and_band(common_area)
                    value, low, high = contrast(ref, other)
                    common.append(dict(f_prop=force, epsilon=epsilon, phi_area=common_area,
                        speed_reference=ref[0], speed_interpolated=other[0], delta_speed_percent=value,
                        sensitivity_low_pct=low, sensitivity_high_pct=high))
    return pd.DataFrame(rows), pd.DataFrame(common)


def endpoints(data):
    rows = []
    for (force, epsilon), curve in data.groupby(["f_prop", "epsilon"]):
        curve = curve.sort_values("phi_nominal")
        first, last = curve.iloc[0], curve.iloc[-1]
        for field in METRICS:
            y = curve[field].to_numpy()
            se = curve[field + "_ep_max"].to_numpy()
            delta = y[-1] - y[0]
            margin = 2 * (se[-1] + se[0])
            adjacent = np.abs(np.diff(y)) > 2 * (se[1:] + se[:-1])
            rows.append(dict(f_prop=force, epsilon=epsilon, quantity=field,
                phi_first=first["phi_nominal"], phi_last=last["phi_nominal"],
                first=y[0], last=y[-1], difference=delta,
                change_percent=100 * delta / y[0] if y[0] > 0 else np.nan,
                sensitivity_low=delta - margin, sensitivity_high=delta + margin,
                endpoint_separated=abs(delta) > margin,
                adjacent_separated=int(adjacent.sum()), adjacent_total=len(adjacent)))
    return pd.DataFrame(rows)


def plot_metric(ax, rows, x, y, *, linestyle="-", scale=1.):
    for eps, group in rows.sort_values(x).groupby("epsilon"):
        ax.errorbar(group[x], scale * group[y], yerr=scale * group[y + "_ep_max"],
                    color=COLORS[eps], marker=MARKERS[eps], linestyle=linestyle,
                    markersize=2.9, linewidth=1.15, elinewidth=.7, capsize=1.4)


def epsilon_handles():
    return [Line2D([], [], color=color, marker=MARKERS[eps], markersize=4,
                   linewidth=1.25, label=rf"$\epsilon={eps:g}$")
            for eps, color in COLORS.items()]


def label_panels(axes):
    for i, ax in enumerate(axes.flat):
        ax.text(.008, 1.025, f"({chr(97 + i)})", transform=ax.transAxes,
                va="bottom", ha="left", weight="bold", fontsize=9)


def save_figure(fig, folder, stem):
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{stem}.{extension}", dpi=210, facecolor="white")
    plt.close(fig)


def figures(data, matched, common_area, tau, folder):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "figure.dpi": 120})
    forces = sorted(data["f_prop"].unique())
    style = [Line2D([], [], color=".25", ls="-", label=r"$\lambda_{\min}$ e $P$"),
             Line2D([], [], color=".25", ls="--", label=r"$\lambda_{\max}$ e $S$")]
    fig, axes = plt.subplots(3, len(forces), sharex=True, sharey="row", figsize=(11, 7.4))
    fig.subplots_adjust(left=.092, right=.98, top=.83, bottom=.105, wspace=.16, hspace=.19)
    for col, force in enumerate(forces):
        rows = data[data["f_prop"] == force]
        a, b, c = axes[:, col]
        a.set_title(rf"$f_{{\mathrm{{prop}}}}={force:g}$", fontsize=12, pad=9)
        plot_metric(a, rows, "phi_nominal", "lambda_min")
        plot_metric(a, rows, "phi_nominal", "lambda_max", linestyle="--")
        plot_metric(b, rows, "phi_nominal", "P")
        plot_metric(b, rows, "phi_nominal", "S", linestyle="--")
        plot_metric(c, rows, "phi_nominal", "speed_per_force", scale=1e3)
        c.set_xlabel(r"$\phi$ nominal")
    axes[0, 0].set_ylabel(r"$\langle\lambda_{\min,\max}\rangle$")
    axes[1, 0].set_ylabel(r"Ordem interna $P, S$")
    axes[2, 0].set_ylabel(r"$10^3\,\langle v\rangle/\langle F_{\rm ativa}\rangle$")
    label_panels(axes)
    fig.suptitle("Deformação, ordem interna e translação", weight="bold", y=.978)
    fig.legend(handles=epsilon_handles() + style, loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(.52, .94), fontsize=9)
    fig.text(.5, .029, f"Barras mostram o maior EP temporal entre as cinco partições. "
                       f"Atraso da velocidade = {tau:g}.", ha="center", fontsize=9)
    save_figure(fig, folder, "fig1_forma_ordem_transporte")

    fig, axes = plt.subplots(2, len(forces), figsize=(11, 6.15))
    fig.subplots_adjust(left=.095, right=.98, top=.80, bottom=.125, wspace=.31, hspace=.40)
    for col, force in enumerate(forces):
        top, bottom = axes[:, col]
        rows = data[data["f_prop"] == force]
        plot_metric(top, rows, "phi_area", "speed")
        top.axvline(common_area, color=".62", lw=.8, ls=":")
        top.set(xlabel=r"Fração pelos contornos $\phi_A$",
                title=rf"$f_{{\mathrm{{prop}}}}={force:g}$")
        if col == 0:
            top.set_ylabel(r"Velocidade $\langle v\rangle$")
        for eps, selected in matched[matched["f_prop"] == force].groupby("epsilon"):
            for kind, linestyle in (("nominal", "--"), ("contours", "-")):
                curve = selected[selected["coverage_type"] == kind].sort_values("phi_nominal_reference")
                x = curve["phi_nominal_reference"].to_numpy()
                bottom.plot(x, curve["delta_speed_percent"], linestyle, lw=1.35, color=COLORS[eps])
                if kind == "contours":
                    bottom.fill_between(x, curve["sensitivity_low_pct"], curve["sensitivity_high_pct"],
                                        color=COLORS[eps], alpha=.16, linewidth=0)
        bottom.axhline(0, color=".54", lw=.8)
        bottom.set(xlabel=r"$\phi$ nominal da referência ($\epsilon=0$)", ylim=(-32, 10))
    axes[1, 0].set_ylabel(r"Variação relativa a $\epsilon=0$ (%)")
    label_panels(axes)
    fig.suptitle("Contraste de velocidade ao igualar a área ocupada", weight="bold", y=.98)
    kinds = [Line2D([], [], color=".25", ls="--", label=r"Mesmo $\phi$ nominal"),
             Line2D([], [], color=".25", ls="-", label=r"Mesma $\phi_A$")]
    fig.legend(handles=epsilon_handles() + kinds, loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(.53, .932), fontsize=9)
    fig.text(.5, .030, rf"Linha vertical em $\phi_A={common_area:g}$. "
                       "Faixas de sensibilidade calculadas com ±2 EP máximos.",
             ha="center", fontsize=9)
    save_figure(fig, folder, "fig2_efeito_epsilon")

    fig, axes = plt.subplots(2, len(forces), sharex=True, sharey="row", figsize=(11, 5.85))
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.12, wspace=.15, hspace=.21)
    for col, force in enumerate(forces):
        rows = data[data["f_prop"] == force]
        top, bottom = axes[:, col]
        top.set_title(rf"$f_{{\mathrm{{prop}}}}={force:g}$", fontsize=12, pad=9)
        plot_metric(top, rows, "phi_nominal", "C_p")
        top.axhline(0, color=".55", lw=.8)
        plot_metric(bottom, rows, "phi_nominal", "v_parallel", scale=tau)
        bottom.set_xlabel(r"$\phi$ nominal")
    axes[0, 0].set_ylabel(rf"$C_p({tau:g})$")
    axes[1, 0].set_ylabel(rf"$L_\parallel({tau:g})={tau:g}\,V_\parallel({tau:g})$")
    label_panels(axes)
    fig.suptitle("Memória orientacional e deslocamento projetado", weight="bold", y=.975)
    fig.legend(handles=epsilon_handles(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.52, .924))
    fig.text(.5, .026, f"Primeiro atraso amostrado = {tau:g}. Barras mostram o maior EP temporal.",
             ha="center", fontsize=9)
    save_figure(fig, folder, "fig3_direcionalidade")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pasta ou ZIP com os CSVs das comparações.")
    parser.add_argument("--out", type=Path, default=Path("analise_celulas_campanha"))
    parser.add_argument("--area-comum", type=float, default=.55)
    parser.add_argument("--no-plots", action="store_true", help="Gera somente as tabelas.")
    args = parser.parse_args()
    runs, conditions, blocks, prior, dynamics = read_comparisons(args.source)
    validate(runs, conditions, blocks, dynamics)
    block_data = blocks[blocks["blocks_valid"] >= 4]
    maxima = block_data.pivot_table(index="run", columns="quantity", values="sem_blocks",
                                   aggfunc="max").add_suffix("_ep_max")
    data = runs.join(maxima, on="run").sort_values(KEY)
    matched, common = matched_rows(data, args.area_comum)
    if not np.isfinite(common["delta_speed_percent"]).all():
        print(f"Algumas curvas não cobrem a área de referência {args.area_comum:g}. "
              "As comparações nessa área ficam vazias quando exigiriam extrapolação.")
    keys = ["f_prop", "epsilon", "coverage_type", "phi_nominal_reference"]
    check = matched.merge(prior, on=keys, validate="one_to_one")
    if len(check) != len(prior) or len(matched) != len(prior):
        raise ValueError("A comparação não reproduziu a cobertura da tabela original.")
    np.testing.assert_allclose(check["delta_speed_percent_x"], check["delta_speed_percent_y"],
                               atol=1e-9)

    tables = args.out / "tabelas"
    tables.mkdir(parents=True, exist_ok=True)
    tau = runs["speed_tau"].iloc[0]
    dyn = dynamics.copy()
    dyn["L_parallel"] = dyn["tau"] * dyn["v_parallel"]
    dyn["L_parallel_sem_blocks"] = dyn["tau"] * dyn["v_parallel_sem_blocks"]
    # O primeiro atraso é o atraso de comparação em toda esta campanha.
    dyn = dyn.join(dyn.loc[dyn["tau"] == tau].set_index("run")["L_parallel"]
                   .rename("L_at_speed_tau"), on="run")
    dyn["L_parallel_ratio"] = dyn["L_parallel"] / dyn["L_at_speed_tau"]
    outputs = {
        "condicoes_ep_temporal": data.drop(columns=["trajectory", "input_file"]),
        "extremos_densidade": endpoints(data),
        "contrastes_em_area_e_phi": matched,
        "contrastes_area_comum": common,
        "ep_mediano_por_blocos": block_data.groupby(["quantity", "blocks_requested", "block_duration"],
                                                   as_index=False)["sem_blocks"].median(),
        "deslocamento_projetado": dyn,
    }
    for name, table in outputs.items():
        table.to_csv(tables / f"{name}.csv", index=False)
    if not args.no_plots:
        folder = args.out / "figuras"
        folder.mkdir(parents=True, exist_ok=True)
        figures(data, matched, args.area_comum, tau, folder)
    print(f"{len(data)} condições e {len(matched)} contrastes reproduzidos.")
    print(common.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
