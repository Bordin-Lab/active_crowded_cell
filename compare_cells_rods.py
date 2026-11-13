#!/usr/bin/env python3
"""Compara forma, polarização e translação usando a saída de analyze_cells_rods.py.

Exemplo:
    python compare_cells_rods.py --root analise_campanha --out comparacoes

Use --root mais de uma vez para reunir análises de réplicas existentes.
Os arquivos cells_frames.csv.gz e summary.json devem ficar juntos.
O programa usa NumPy e, para figuras, Matplotlib. --no-plots gera apenas CSVs.

Todos os atrasos e --start-time usam as unidades físicas da coluna time.
Por padrão, a velocidade é comparada no maior intervalo entre frames.
Se as cadências não forem compatíveis, indique um múltiplo comum com
--speed-lag. O programa usa pares observados, sem interpolar no tempo.

Cp é <p(t).p(t+tau)> dividido pela média geométrica das potências de p
nos dois extremos dos mesmos pares. C_direction usa os versores de p.
v_parallel projeta o deslocamento por unidade de tempo no versor inicial.
speed_per_force divide a velocidade pelo módulo médio de f_prop*N_rods*p
no início dos pares usados. É uma razão cinemática para o atraso escolhido.
Em cada atraso, o programa calcula a média sobre células para cada origem
de tempo e então dá o mesmo peso às origens válidas.

phi_area é a soma das áreas internas dos contornos dividida pela área da
caixa. Ela exige um frame com todas as células presentes. Se os contornos
se sobrepõem, a área comum é contada para cada célula.
matched_coverage.csv interpola a velocidade média de cada epsilon nas
abscissas da referência, dentro do domínio medido. A comparação é
descritiva e inclui a fração nominal e a fração calculada pelos contornos.

Cada réplica tem peso igual na média da condição. As estimativas temporais
mantêm as células de cada frame juntas e usam blocos contíguos. Para os
deslocamentos de um bloco, os dois extremos devem estar nele. O CSV de
blocos permite verificar a sensibilidade do erro padrão à duração do bloco.
As figuras usam EP entre réplicas quando há réplicas e EP temporal para
pontos com uma única trajetória. Um EP temporal requer ao menos quatro
blocos válidos e depende da estacionariedade e da correlação entre blocos.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GEOMETRY = ("phi_area", "area", "Rg2_ring", "asphericity", "lambda_max",
            "lambda_min", "P", "S", "rods_per_area", "rods_shape_cos2")
DYNAMICS = ("C_p", "C_direction", "speed", "v_parallel", "speed_per_force")
COMPARISON_METRICS = GEOMETRY + DYNAMICS
FRAME_FIELDS = (
    "frame", "time", "cell_id", "n_rods", "com_x_unwrapped", "com_y_unwrapped",
    "polar_x", "polar_y", "area", "Rg2_ring", "asphericity", "lambda_max",
    "lambda_min", "S", "rods_shape_cos2",
)


@dataclass
class Run:
    path: Path
    source: str
    force: float | None
    epsilon: float | None
    phi: float | None
    stride: float
    n_cells: int
    box_area: float

    def labels(self):
        return dict(f_prop=self.force, epsilon=self.epsilon, phi_nominal=self.phi)


def number(value):
    if value is None or value == "":
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def find_runs(roots):
    files = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Pasta inexistente: {root}")
        for name in ("cells_frames.csv", "cells_frames.csv.gz"):
            for path in root.rglob(name):
                files[path.parent.resolve()] = path.resolve()
    if not files:
        raise ValueError("Nenhum cells_frames.csv.gz ou cells_frames.csv nas pastas.")
    runs = []
    sources = set()
    for path in sorted(files.values()):
        meta = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))["summary"]
        stride = number(meta["frame_dt_median"])
        box_area = float(meta["box_Lx"]) * float(meta["box_Ly"])
        n_cells = int(meta["n_cells"])
        if stride is None or stride <= 0 or not 0 < box_area < np.inf or n_cells < 1:
            raise ValueError(f"Cadência, caixa ou número de células inválido em {path.parent}")
        source = meta.get("trajectory") or str(path)
        if source in sources:
            raise ValueError(f"A mesma trajetória aparece mais de uma vez: {source}")
        sources.add(source)
        runs.append(Run(path, source, number(meta.get("f_prop")), number(meta.get("epsilon")),
                        number(meta.get("phi")), stride, n_cells, box_area))
    return runs


def read_frames(run, start_time):
    opener = gzip.open if run.path.suffix == ".gz" else open
    with opener(run.path, "rt", encoding="utf-8") as f:
        header = next(csv.reader(f))
        missing = sorted(set(FRAME_FIELDS) - set(header))
        if missing:
            raise ValueError("Colunas ausentes: " + ", ".join(missing))
        data = np.loadtxt(f, delimiter=",", usecols=[header.index(k) for k in FRAME_FIELDS], ndmin=2)
    if len(data) == 0 or not np.isfinite(data[:, :3]).all():
        raise ValueError("Frames, tempos ou identificadores ausentes.")
    if start_time is not None:
        data = data[data[:, 1] >= start_time]
    if len(data) == 0:
        raise ValueError("Nenhum frame a partir de --start-time.")
    frames = data[:, 0].astype(np.int64)
    if np.any(frames != data[:, 0]) or np.any(data[:, 2] != np.rint(data[:, 2])):
        raise ValueError("Os índices de frame e célula devem ser inteiros.")
    frames -= frames.min()
    time0 = float(data[np.argmin(frames), 1])
    if not np.allclose(data[:, 1], time0 + frames * run.stride,
                       rtol=0, atol=max(1e-9, 1e-6 * run.stride)):
        raise ValueError("Os tempos não correspondem a uma cadência uniforme.")
    _, cell_index = np.unique(data[:, 2], return_inverse=True)
    n_observed = int(cell_index.max()) + 1
    if n_observed > run.n_cells:
        raise ValueError("Há mais IDs celulares que o total em summary.json.")
    n_frames = int(frames.max()) + 1
    flat_index = frames * n_observed + cell_index
    if len(np.unique(flat_index)) != len(flat_index):
        raise ValueError("Há linhas duplicadas para a mesma célula e frame.")
    grid = np.full((n_frames, n_observed, len(FRAME_FIELDS) - 3), np.nan)
    grid[frames, cell_index] = data[:, 3:]
    return time0, {key: grid[:, :, j] for j, key in enumerate(FRAME_FIELDS[3:])}


def mean(values, axis=None):
    valid = np.isfinite(values)
    total = np.sum(values, where=valid, axis=axis)
    count = np.sum(valid, axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def sem(values, minimum=2):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < minimum:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def geometry_series(grid, run, polar_magnitude):
    area = grid["area"]
    complete = np.isfinite(area).sum(axis=1) == run.n_cells
    phi_area = np.where(complete, np.sum(area, axis=1) / run.box_area, np.nan)
    rods_per_area = np.divide(grid["n_rods"], area, out=np.full_like(area, np.nan),
                              where=area > 0)
    series = {key: mean(grid[key], axis=1) for key in
              ("area", "Rg2_ring", "asphericity", "lambda_max", "lambda_min", "S", "rods_shape_cos2")}
    series.update(phi_area=phi_area, P=mean(polar_magnitude, axis=1),
                  rods_per_area=mean(rods_per_area, axis=1))
    return series


def lag_series(xy, p, magnitude, rods, missing_prefix, lag, stride, force):
    """Calcula médias celulares por origem de tempo e evita ligar lacunas do COM."""
    n = len(xy) - lag
    left, right = slice(0, n), slice(lag, None)
    present = (missing_prefix[lag + 1:] - missing_prefix[:n]) == 0
    p0, p1 = p[left], p[right]
    m0, m1 = magnitude[left], magnitude[right]
    both = present & np.isfinite(m0) & np.isfinite(m1)
    dot = np.einsum("tci,tci->tc", p0, p1)
    nonzero = both & (m0 > 1e-12) & (m1 > 1e-12)
    direction = np.divide(dot, m0 * m1, out=np.full_like(dot, np.nan), where=nonzero)
    series = {
        "dot": mean(np.where(both, dot, np.nan), axis=1),
        "power0": mean(np.where(both, m0**2, np.nan), axis=1),
        "power1": mean(np.where(both, m1**2, np.nan), axis=1),
        "C_direction": mean(direction, axis=1),
    }
    if lag:
        velocity = (xy[right] - xy[left]) / (lag * stride)
        speed = np.linalg.norm(velocity, axis=-1)
        projection = np.einsum("tci,tci->tc", velocity, p0)
        v_parallel = np.divide(projection, m0, out=np.full_like(projection, np.nan),
                               where=present & (m0 > 1e-12))
        series["speed"] = mean(np.where(present, speed, np.nan), axis=1)
        series["v_parallel"] = mean(v_parallel, axis=1)
        if force is not None and force > 0:
            active = force * rods[left] * m0
            force_mask = present & np.isfinite(active)
            series["mean_force"] = mean(np.where(force_mask, active, np.nan), axis=1)
            series["speed_for_force"] = mean(np.where(force_mask, speed, np.nan), axis=1)
    counts = dict(n_pairs=int(present.sum()), n_polar_pairs=int(both.sum()),
                  n_direction_pairs=int(nonzero.sum()), n_time_origins=int(present.any(axis=1).sum()))
    return series, counts


def lag_means(series, section):
    values = {key: float(mean(x[section])) for key, x in series.items()}
    norm = math.sqrt(values["power0"] * values["power1"])
    result = dict(C_p=values["dot"] / norm if norm > 0 else np.nan,
                  C_direction=values["C_direction"], speed=values.get("speed", np.nan),
                  v_parallel=values.get("v_parallel", np.nan), speed_per_force=np.nan)
    if values.get("mean_force", 0) > 0:
        result["speed_per_force"] = values["speed_for_force"] / values["mean_force"]
    return result


def block_ranges(n_frames, n_blocks):
    cuts = np.linspace(0, n_frames, n_blocks + 1, dtype=int)
    return list(zip(cuts[:-1], cuts[1:]))


def persistence_time(curve, field):
    data = [(r["tau"], r[field]) for r in curve if r["tau"] > 0 and np.isfinite(r[field])]
    if not data:
        return np.nan, "sem_dados"
    threshold = math.exp(-1)
    if data[0][1] <= threshold:
        return np.nan, "abaixo_do_limiar_no_primeiro_atraso"
    for (t0, c0), (t1, c1) in zip(data[:-1], data[1:]):
        if c1 <= threshold:
            return t0 + (t1 - t0) * (c0 - threshold) / (c0 - c1), "cruzamento_interpolado"
    return np.nan, "sem_cruzamento_na_janela"


def lag_index(tau, stride):
    lag = round(tau / stride)
    if lag < 1 or not math.isclose(lag * stride, tau, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"Atraso {tau:g} incompatível com intervalo {stride:g}. "
                         "Use --speed-lag com um múltiplo comum.")
    return lag


def analyze_run(run, run_id, args, speed_tau):
    time0, grid = read_frames(run, args.start_time)
    n_frames = len(grid["area"])
    k_speed = lag_index(speed_tau, run.stride)
    if k_speed >= n_frames:
        raise ValueError("A trajetória é curta para o atraso escolhido.")
    xy = np.stack((grid["com_x_unwrapped"], grid["com_y_unwrapped"]), axis=-1)
    p = np.stack((grid["polar_x"], grid["polar_y"]), axis=-1)
    magnitude = np.linalg.norm(p, axis=-1)
    missing = ~np.isfinite(xy).all(axis=-1)
    prefix = np.vstack((np.zeros(xy.shape[1], dtype=int), np.cumsum(missing, axis=0)))
    shape = geometry_series(grid, run, magnitude)
    result = dict(run=run_id, trajectory=run.source, input_file=str(run.path), **run.labels(),
                  n_frames=n_frames, n_cells=run.n_cells, time_first=time0, frame_dt=run.stride,
                  speed_tau=speed_tau, n_complete_area_frames=int(np.isfinite(shape["phi_area"]).sum()))
    result.update({key: float(mean(x)) for key, x in shape.items()})

    # Atrasos curtos completos e malha geométrica quando a série é muito longa.
    cap = (n_frames - 1) // 4 if args.max_lag is None else int(args.max_lag / run.stride + 1e-9)
    cap = min(n_frames - 1, max(1, cap))
    if cap <= 80:
        lags = set(range(cap + 1))
    else:
        lags = set(range(21)) | set(np.rint(np.geomspace(21, cap, 48)).astype(int))
    lags.add(k_speed)
    blocks = block_ranges(n_frames, args.blocks)
    curve = []
    for lag in sorted(lags):
        series, counts = lag_series(xy, p, magnitude, grid["n_rods"], prefix, lag,
                                    run.stride, run.force)
        estimates = lag_means(series, slice(None))
        row = dict(run=run_id, **run.labels(), tau=lag * run.stride, lag_frames=lag,
                   **counts, **estimates)
        by_block = [lag_means(series, slice(a, b - lag)) for a, b in blocks if b - a - lag >= 2]
        for key in DYNAMICS:
            row[key + "_sem_blocks"] = sem([m[key] for m in by_block], minimum=4)
        curve.append(row)
        if lag == k_speed:
            result.update(estimates)
            speed_series = series
        if lag == 1:
            result["C_p_first_lag"] = estimates["C_p"]
            result["C_direction_first_lag"] = estimates["C_direction"]
    result["tau_p_1e"], result["tau_p_status"] = persistence_time(curve, "C_p")
    result["tau_direction_1e"], result["tau_direction_status"] = persistence_time(curve, "C_direction")

    block_rows = []
    for n_blocks in sorted({4, 8, 12, 16, args.blocks}):
        estimates = defaultdict(list)
        lengths = []
        for a, b in block_ranges(n_frames, n_blocks):
            if b - a < 2:
                continue
            lengths.append(b - a)
            for key, values in shape.items():
                estimates[key].append(float(mean(values[a:b])))
            if b - a - k_speed >= 2:
                for key, value in lag_means(speed_series, slice(a, b - k_speed)).items():
                    estimates[key].append(value)
        for key in COMPARISON_METRICS:
            values = estimates[key]
            error = sem(values, minimum=4)
            block_rows.append(dict(run=run_id, **run.labels(), quantity=key,
                blocks_requested=n_blocks, blocks_valid=int(np.isfinite(values).sum()),
                block_duration=float(mean(lengths)) * run.stride,
                mean_of_blocks=float(mean(values)), sem_blocks=error))
            if n_blocks == args.blocks:
                result[key + "_sem_blocks"] = error
    return result, curve, block_rows


def condition_key(row):
    return row["f_prop"], row["epsilon"], row["phi_nominal"]


def aggregate(rows, metrics, dynamic=False):
    groups = defaultdict(list)
    for row in rows:
        condition = condition_key(row)
        if any(v is None for v in condition):
            continue
        key = condition + (float(f'{row["tau"]:.10g}'),) if dynamic else condition
        groups[key].append(row)
    output = []
    for key, members in sorted(groups.items()):
        out = dict(zip(("f_prop", "epsilon", "phi_nominal"), key[:3]))
        if dynamic:
            out["tau"] = key[3]
        out["n_trajectories"] = len(members)
        for metric in metrics:
            valid = [r for r in members if np.isfinite(r[metric])]
            values = [r[metric] for r in valid]
            out[metric + "_mean"] = float(mean(values))
            out[metric + "_n_trajectories"] = len(valid)
            out[metric + "_sem_runs"] = sem(values)
            out[metric + "_sem_blocks_single"] = (
                valid[0][metric + "_sem_blocks"] if len(valid) == 1 else np.nan)
        output.append(out)
    return output


def matched_speeds(conditions, reference_eps):
    output = []
    for force in sorted({r["f_prop"] for r in conditions}):
        subset = [r for r in conditions if r["f_prop"] == force]
        reference = [r for r in subset if r["epsilon"] == reference_eps]
        for eps in sorted({r["epsilon"] for r in subset} - {reference_eps}):
            other = [r for r in subset if r["epsilon"] == eps]
            for field, label in (("phi_nominal", "nominal"), ("phi_area_mean", "contours")):
                curve = defaultdict(list)
                for row in other:
                    if np.isfinite(row[field]) and np.isfinite(row["speed_mean"]):
                        curve[row[field]].append(row["speed_mean"])
                if not curve:
                    continue
                x = np.array(sorted(curve))
                y = np.array([mean(curve[k]) for k in x])
                for row in sorted(reference, key=lambda r: r["phi_nominal"]):
                    target = row[field]
                    v0 = row["speed_mean"]
                    if not (np.isfinite(v0) and x[0] <= target <= x[-1]):
                        continue
                    v = float(np.interp(target, x, y))
                    output.append(dict(f_prop=force, epsilon_reference=reference_eps, epsilon=eps,
                        coverage_type=label, coverage=target,
                        phi_nominal_reference=row["phi_nominal"], speed_reference=v0,
                        speed_interpolated=v, delta_speed=v-v0,
                        delta_speed_percent=100 * (v / v0 - 1) if v0 > 0 else np.nan))
    return output


def save_csv(path, rows, empty_header):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or empty_header.split(",")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                             else v for k, v in row.items()})


def plot_line(ax, rows, x_key, metric, label, color):
    valid = sorted((r for r in rows if np.isfinite(r[x_key]) and np.isfinite(r[metric + "_mean"])),
                   key=lambda r: r[x_key])
    if not valid:
        return
    x = np.array([r[x_key] for r in valid])
    y = np.array([r[metric + "_mean"] for r in valid])
    ax.plot(x, y, "o-", ms=3, lw=1.3, color=color, label=label)
    for key, alpha in (("_sem_runs", .9), ("_sem_blocks_single", .45)):
        errors = np.array([r[metric + key] for r in valid])
        good = np.isfinite(errors)
        if good.any():
            ax.errorbar(x[good], y[good], yerr=errors[good], fmt="none",
                        ecolor=color, alpha=alpha, lw=1, capsize=2)


def plot_results(out, conditions, dynamics, speed_tau):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error_note = "Barras de EP. Mais claras para estimativa temporal em uma trajetória."
    panels = (
        ("phi_nominal", "speed", r"$\phi$ nominal", "Velocidade"),
        ("phi_area_mean", "speed", r"$\phi_A$ pelos contornos", "Velocidade"),
        ("phi_area_mean", "speed_per_force", r"$\phi_A$ pelos contornos", "Velocidade / força ativa"),
        ("phi_nominal", "lambda_max", r"$\phi$ nominal", r"$\langle\lambda_{\max}\rangle$"),
        ("phi_nominal", "lambda_min", r"$\phi$ nominal", r"$\langle\lambda_{\min}\rangle$"),
        ("phi_nominal", "P", r"$\phi$ nominal", "Polarização interna"),
        ("phi_nominal", "S", r"$\phi$ nominal", "Ordem nemática interna"),
        ("rods_per_area_mean", "P", r"$\langle N_r/A\rangle$", "Polarização interna"),
        ("rods_per_area_mean", "S", r"$\langle N_r/A\rangle$", "Ordem nemática interna"),
    )
    for force in sorted({r["f_prop"] for r in conditions}):
        static = [r for r in conditions if r["f_prop"] == force]
        eps_values = sorted({r["epsilon"] for r in static})
        colors = {eps: plt.get_cmap("tab10")(i % 10) for i, eps in enumerate(eps_values)}
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
        for ax, (x_key, metric, xlabel, ylabel) in zip(axes.flat, panels):
            for eps in eps_values:
                plot_line(ax, [r for r in static if r["epsilon"] == eps], x_key,
                          metric, f"ε = {eps:g}", colors[eps])
            ax.set(xlabel=xlabel, ylabel=ylabel)
        axes[0, 0].legend()
        fig.suptitle(f"Força {force:g} · intervalo de velocidade {speed_tau:g}\n{error_note}")
        fig.savefig(out / f"comparacoes_f{force:g}.png", dpi=180)
        plt.close(fig)

        phi = sorted({r["phi_nominal"] for r in static})
        extremes = sorted({phi[0], phi[-1]})
        fig, axes = plt.subplots(3, len(extremes), squeeze=False,
                                 figsize=(5.5 * len(extremes), 8), constrained_layout=True)
        for col, density in enumerate(extremes):
            for j, (key, label) in enumerate((("C_p", r"$C_p$"), ("C_direction", r"$C_{\hat p}$"),
                                               ("v_parallel", r"$v_\parallel$"))):
                ax = axes[j, col]
                for eps in eps_values:
                    rows = [r for r in dynamics if r["f_prop"] == force and r["epsilon"] == eps
                            and r["phi_nominal"] == density]
                    plot_line(ax, rows, "tau", key, f"ε = {eps:g}", colors[eps])
                ax.axhline(1 / math.e if j < 2 else 0, color=".65", lw=.8, ls=":")
                ax.set(xlabel="Atraso", ylabel=label)
            axes[0, col].set_title(f"φ nominal = {density:g}")
            axes[0, col].legend()
        fig.suptitle(f"Persistência e movimento na direção polar · força {force:g}\n{error_note}")
        fig.savefig(out / f"persistencia_f{force:g}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", type=Path, required=True,
                        help="Pasta de saída do analisador. Pode repetir.")
    parser.add_argument("--out", type=Path, default=Path("comparacoes_celulas"))
    parser.add_argument("--start-time", type=float, help="Primeiro tempo físico a incluir.")
    parser.add_argument("--speed-lag", type=float, help="Atraso físico comum para medir a velocidade.")
    parser.add_argument("--max-lag", type=float, help="Janela da curva. Padrão: um quarto da duração.")
    parser.add_argument("--blocks", type=int, default=6, help="Número de blocos do EP temporal principal.")
    parser.add_argument("--reference-eps", type=float, default=0.)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    for key in ("start_time", "speed_lag", "max_lag", "reference_eps"):
        value = getattr(args, key)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{key.replace('_', '-')} precisa ser finito")
    if args.blocks < 4 or (args.max_lag is not None and args.max_lag <= 0):
        parser.error("Use pelo menos quatro blocos e uma janela de atraso positiva.")

    runs = find_runs(args.root)
    speed_tau = max(r.stride for r in runs) if args.speed_lag is None else args.speed_lag
    for run in runs:
        lag_index(speed_tau, run.stride)
    summaries, curves, block_rows = [], [], []
    for index, run in enumerate(runs, 1):
        run_id = f"run_{index:04d}"
        try:
            summary, curve, blocks = analyze_run(run, run_id, args, speed_tau)
        except ValueError as exc:
            raise ValueError(f"{run.path}: {exc}") from exc
        summaries.append(summary)
        curves.extend(curve)
        block_rows.extend(blocks)
        cp = summary["C_p_first_lag"]
        print(f"[{index}/{len(runs)}] {run.path.parent.name} | "
              f"φ_A={summary['phi_area']:.4g} | C_p({run.stride:g})={cp:.4g}", flush=True)
        if any(v is None for v in condition_key(summary)):
            print(f"[AVISO] {run_id}: metadados incompletos. A série individual foi mantida.", file=sys.stderr)
    conditions = aggregate(summaries, COMPARISON_METRICS)
    dynamics = aggregate(curves, DYNAMICS, dynamic=True)
    matched = matched_speeds(conditions, args.reference_eps)
    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out / "runs_summary.csv", summaries, "run")
    save_csv(args.out / "orientation_dynamics.csv", curves, "run,tau")
    save_csv(args.out / "block_statistics.csv", block_rows, "run,quantity")
    save_csv(args.out / "conditions_summary.csv", conditions, "f_prop,epsilon,phi_nominal,n_trajectories")
    save_csv(args.out / "orientation_by_condition.csv", dynamics, "f_prop,epsilon,phi_nominal,tau")
    save_csv(args.out / "matched_coverage.csv", matched,
             "f_prop,epsilon_reference,epsilon,coverage_type,coverage,speed_reference,speed_interpolated,delta_speed,delta_speed_percent")
    if not args.no_plots and conditions:
        plot_results(args.out, conditions, dynamics, speed_tau)
    print(f"{len(summaries)} trajetórias | {len(conditions)} condições | "
          f"{len(matched)} comparações na faixa comum | {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        sys.exit(f"Erro: {exc}")
