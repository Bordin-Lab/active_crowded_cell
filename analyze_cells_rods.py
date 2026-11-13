#!/usr/bin/env python3
"""Forma de células de membrana + orientação de bastonetes ativos em 2D.

Analisa LAMMPS dumps ortogonais do gerador cell_phi.py, sem ESPResSo,
pandas, TrajPy ou classificadores de fases. NumPy é obrigatório;
Matplotlib só é necessário para criar figuras. Ver README.md para as
definições, unidades e exemplos de análise recursiva da campanha.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = "1.0.0"


@dataclass
class Box:
    lo: np.ndarray
    length: np.ndarray
    periodic: np.ndarray

    def delta(self, d):
        """Vetor de mínima imagem; mantém eixos não periódicos intactos."""
        d = np.asarray(d, dtype=float).copy()
        d[..., self.periodic] -= self.length[self.periodic] * np.rint(
            d[..., self.periodic] / self.length[self.periodic])
        return d

    def fold(self, x):
        x = np.asarray(x, dtype=float).copy()
        p = self.periodic
        x[..., p] = self.lo[p] + (x[..., p] - self.lo[p]) % self.length[p]
        return x


def read_frames(path):
    """Lê um frame por vez: memória não cresce com o arquivo de partículas.

    Os inteiros são representados exatamente até 2**53 no array float64.
    Dumps triclinicos são rejeitados; x/y ou xs/ys são aceitos.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError("Cabeçalho TIMESTEP ausente ou arquivo truncado")
            step = int(f.readline())
            if f.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Cabeçalho NUMBER OF ATOMS ausente")
            n = int(f.readline())
            bounds = f.readline().split()
            if bounds[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("Cabeçalho BOX BOUNDS ausente")
            if any(v in bounds[3:] for v in ("xy", "xz", "yz")):
                raise ValueError("Caixa triclinica: este analisador aceita caixas ortogonais")
            b = [list(map(float, f.readline().split())) for _ in range(3)]
            lo = np.array([b[0][0], b[1][0]])
            length = np.array([b[0][1]-b[0][0], b[1][1]-b[1][0]])
            if np.any(length <= 0):
                raise ValueError("Comprimento da caixa deve ser positivo")
            flags = bounds[3:]
            periodic = np.array([v == "pp" for v in flags[:2]]) if len(flags) >= 2 else np.ones(2, bool)
            box = Box(lo, length, periodic)
            h = f.readline().split()
            if h[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Cabeçalho ATOMS ausente")
            columns = h[2:]
            lines = list(itertools.islice(f, n))
            if len(lines) != n:
                raise ValueError("Frame incompleto: faltam partículas")
            a = np.loadtxt(lines, ndmin=2)
            if a.shape != (n, len(columns)):
                raise ValueError("Número de colunas incompatível com o cabeçalho")
            for key in ("id", "mol", "cell", "type"):
                if key not in columns:
                    raise ValueError("Coluna obrigatória ausente: " + key)
            c = {k: a[:, columns.index(k)] for k in ("id", "mol", "cell", "type")}
            for pair, scaled in ((("x", "y"), False), (("xu", "yu"), False), (("xs", "ys"), True)):
                if all(k in columns for k in pair):
                    xy = a[:, [columns.index(k) for k in pair]]
                    if scaled:
                        xy = lo + xy * length
                    c["xy"] = box.fold(xy)
                    break
            else:
                raise ValueError("São necessárias coordenadas x/y, xu/yu ou xs/ys")
            yield step, box, c


def groups(keys):
    """Intervalos de chaves já ordenadas; retorna IDs originais."""
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    ends = np.r_[starts[1:], len(keys)]
    for s, e in zip(starts, ends):
        if e > s:
            yield int(keys[s]), slice(s, e)


def ring_geometry(xy, box):
    """Reconstrói o contorno ordenado ANTES de calcular COM e tensor G.

    G_ab = média[(r_a-COM_a)(r_b-COM_b)]. A ordem dos monômeros
    deve seguir o contorno; analyze_one os ordena por id, como no gerador.
    """
    if len(xy) < 3:
        raise ValueError("Membrana com menos de três monômeros")
    bonds = box.delta(np.roll(xy, -1, axis=0) - xy)
    if np.linalg.norm(bonds.sum(axis=0)) > 1e-6 * max(box.length):
        raise ValueError("Contorno não fecha na imagem reconstruída (winding ou ordem incorreta)")
    whole = np.vstack([xy[0], xy[0] + np.cumsum(bonds[:-1], axis=0)])
    center = whole.mean(axis=0)
    r = whole - center
    G = r.T @ r / len(r)
    eig, vec = np.linalg.eigh(G)
    eig = np.maximum(eig, 0)
    rg2 = float(eig.sum())
    b = float((eig[1]-eig[0])**2 / rg2**2) if rg2 > 0 else 0.0
    axis = vec[:, 1]
    shape_theta = float(np.arctan2(axis[1], axis[0]) % np.pi) if b > 1e-8 else np.nan
    perimeter = float(np.linalg.norm(bonds, axis=1).sum())
    area = float(abs(np.sum(r[:, 0]*np.roll(r[:, 1], -1)-r[:, 1]*np.roll(r[:, 0], -1))) / 2)
    q = perimeter / math.sqrt(area) if area > 0 else np.nan
    return dict(com=box.fold(center), whole=whole, center=center,
                Rg2_ring=rg2, Rg_ring=math.sqrt(rg2), asphericity=b,
                lambda_max=float(eig[1]), lambda_min=float(eig[0]),
                shape_theta_rad=shape_theta, perimeter=perimeter, area=area, shape_index_q=q)


def rod_order(u):
    """Polarização P e ordem nemática S em 2D, ambas entre zero e um."""
    if not len(u):
        return dict(P=np.nan, S=np.nan, polar_x=np.nan, polar_y=np.nan,
                    nematic_c=np.nan, nematic_s=np.nan,
                    polar_theta_rad=np.nan, nematic_theta_rad=np.nan)
    p = u.mean(axis=0)
    nc = float(np.mean(u[:, 0]**2-u[:, 1]**2))
    ns = float(np.mean(2*u[:, 0]*u[:, 1]))
    P, S = float(np.linalg.norm(p)), float(np.hypot(nc, ns))
    return dict(P=P, S=S, polar_x=float(p[0]), polar_y=float(p[1]),
                nematic_c=nc, nematic_s=ns,
                polar_theta_rad=float(np.arctan2(p[1], p[0])) if P > 1e-10 else np.nan,
                nematic_theta_rad=float(0.5*np.arctan2(ns, nc)) if S > 1e-10 else np.nan)


def inside_polygon(points, poly):
    """Teste raio-polígono para os centros dos bastonetes, na mesma imagem."""
    if not len(points):
        return np.array([], bool)
    x, y = points[:, 0, None], points[:, 1, None]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crossing_y = (y1 > y) != (y2 > y)
    den = y2-y1
    safe_den = np.where(den != 0, den, 1.0)
    xcross = x1 + (y-y1)*(x2-x1)/safe_den
    return np.logical_xor.reduce(crossing_y & (x < xcross), axis=1)


def rod_groups(c, box, rod_type, rod_beads, reverse=False):
    """Polaridade: menor id -> maior id, igual a apply_rods_activity.

    Mantém cell do arquivo como identidade parental; dentro/fora é medido
    separadamente, sem transferir silenciosamente bastonetes entre células.
    """
    k = np.flatnonzero(c["type"] == rod_type)
    k = k[np.lexsort((c["id"][k], c["mol"][k]))]
    parent = {}
    for rid, sl in groups(c["mol"][k]):
        idx = k[sl]
        if len(idx) != rod_beads:
            raise ValueError(f"Bastonete {rid}: {len(idx)} partículas, esperado {rod_beads}")
        cell_ids = np.unique(c["cell"][idx].astype(int))
        if len(cell_ids) != 1:
            raise ValueError(f"Bastonete {rid} tem mais de uma célula parental")
        xy = c["xy"][idx]
        whole = np.vstack([xy[0], xy[0] + np.cumsum(box.delta(np.diff(xy, axis=0)), axis=0)])
        d = whole[-1]-whole[0]
        length = np.linalg.norm(d)
        if length <= 1e-12:
            raise ValueError(f"Bastonete {rid} de comprimento zero")
        u = d / length * (-1 if reverse else 1)
        parent.setdefault(int(cell_ids[0]), []).append((rid, u, box.fold(whole.mean(axis=0)), float(length)))
    return parent


def msd_series(xy, times, max_fraction=0.25):
    """MSD em tempos físicos; não liga trechos separados por dados ausentes.

    Guarda todos os atrasos solicitados, número de pares e atraso médio real.
    Até T/4 por padrão: evita usar um único par como teste de imobilização.
    """
    n = len(times)
    m = min(n-1, max(1, int((n-1)*max_fraction))) if n > 1 else 0
    if not m:
        return []
    if not np.allclose(np.diff(times), times[1]-times[0], rtol=1e-9, atol=1e-10):
        raise ValueError("MSD requer intervalos temporais uniformes")
    valid = np.isfinite(xy).all(axis=1)
    transitions = np.diff(np.r_[False, valid, False].astype(int))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    accum, counts = np.zeros(m), np.zeros(m, int)
    for begin, end in zip(starts, ends):
        r = xy[begin:end]
        nb = len(r)
        if nb < 2:
            continue
        # Translação não muda o MSD; centralizar reduz cancelamento numérico.
        r = r-r.mean(axis=0)
        fft_n = 1 << ((2*nb-1).bit_length())
        f = np.fft.rfft(r, n=fft_n, axis=0)
        ac = np.fft.irfft(f*np.conjugate(f), n=fft_n, axis=0)[:nb].sum(axis=1)
        cumulative = np.r_[0., np.cumsum(np.sum(r*r, axis=1))]
        lag = np.arange(1, min(m, nb-1)+1)
        numerator = cumulative[nb]-cumulative[lag]+cumulative[nb-lag]-2*ac[lag]
        accum[:len(lag)] += np.maximum(numerator, 0)
        counts[:len(lag)] += nb-lag
    rows = []
    for j in range(m):
        rows.append((j+1, float((j+1)*(times[1]-times[0])),
                     float(accum[j]/counts[j]) if counts[j] else np.nan, int(counts[j])))
    return rows


def fit_msd(rows, tmin, tmax):
    """Expoente efetivo e inclinação linear na janela escolhida; sem rótulo de fase."""
    a = np.array(rows, float)
    result = dict(alpha_eff=np.nan, alpha_log_r2=np.nan, D_slope_window=np.nan,
                  msd_fit_n_lags=0, fit_tau_min=np.nan, fit_tau_max=np.nan)
    if not len(a):
        return result
    ok = (a[:, 1] >= tmin) & (a[:, 1] <= tmax) & (a[:, 2] > 0) & np.isfinite(a[:, 1:3]).all(axis=1)
    if ok.sum() < 5:
        return result
    t, y = a[ok, 1], a[ok, 2]
    slope, intercept = np.polyfit(np.log(t), np.log(y), 1)
    residual = np.log(y) - (slope*np.log(t)+intercept)
    denom = np.sum((np.log(y)-np.log(y).mean())**2)
    d = float(np.polyfit(t, y, 1)[0] / 4.0)
    result.update(alpha_eff=float(slope), alpha_log_r2=float(1-np.sum(residual**2)/denom) if denom > 0 else np.nan,
                  D_slope_window=d, msd_fit_n_lags=int(ok.sum()),
                  fit_tau_min=float(t.min()), fit_tau_max=float(t.max()))
    return result


def finite_mean(x):
    x = np.asarray(x, float)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def finite_std(x):
    x = np.asarray(x, float)
    return float(np.std(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def json_clean(x):
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as f:
        if fieldnames is None:
            fieldnames = list(rows[0]) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


FLOAT_PATTERN = r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_metadata(path, args):
    """Busca nomes do arquivo para fora; suporta f_prop_5/eps_1/traj-phi0.55."""
    labels = {"f_prop": r"(?:f[_-]?prop|force[_-]?prop|propuls(?:ion|ao|ão)|propagacao|prop|force|forca|força|fp|f)",
              "epsilon": r"(?:epsilon|eps|adesao|adesão)", "phi": r"(?:phi|packing)"}
    result = {}
    # Nome mais próximo do arquivo tem prioridade; não lê diretórios superiores ao root.
    parts = [path.name]
    root = Path(args.root).resolve() if args.root else None
    for parent in path.resolve().parents:
        parts.append(parent.name)
        if root and parent == root:
            break
    for key, label in labels.items():
        value = getattr(args, key)
        if value is None:
            rx = re.compile(r"(?:^|[^a-zA-Z])"+label+r"[\s_=:=-]*(?P<v>"+FLOAT_PATTERN+r")", re.I)
            for part in parts:
                match = rx.search(part)
                if match:
                    value = float(match.group("v"))
                    break
        result[key] = value
    # Hierarquia informada pelo usuário: ROOT/força/epsilon/arquivo.
    # Aceita também pastas puramente numéricas, por exemplo ROOT/5/0.0/.
    if root:
        try:
            dirs = path.resolve().relative_to(root).parts[:-1]
        except ValueError:
            dirs = ()
        for key, index in (("f_prop", 0), ("epsilon", 1)):
            if result[key] is None and len(dirs) > index:
                if re.fullmatch(FLOAT_PATTERN, dirs[index]):
                    result[key] = float(dirs[index])
    return result


CELL_COLUMNS = ["frame", "timestep", "time", "cell_id", "n_ring_beads", "n_rods",
    "com_x_wrapped", "com_y_wrapped", "com_x_unwrapped", "com_y_unwrapped",
    "velocity_x", "velocity_y", "speed_interval", "Rg_ring", "Rg2_ring", "asphericity",
    "lambda_max", "lambda_min", "shape_theta_rad", "perimeter", "area", "shape_index_q",
    "rods_density_area", "P", "S", "polar_x", "polar_y", "nematic_c", "nematic_s",
    "polar_theta_rad", "nematic_theta_rad", "rods_shape_cos2", "rods_velocity_cos",
    "polar_velocity_cos", "rods_velocity_cos2", "fraction_rods_inside",
    "active_force_x", "active_force_y", "active_force_magnitude"]
METRICS = ["Rg_ring", "Rg2_ring", "asphericity", "perimeter", "area", "shape_index_q",
    "rods_density_area", "speed_interval", "P", "S", "rods_shape_cos2", "rods_velocity_cos",
    "polar_velocity_cos", "rods_velocity_cos2", "fraction_rods_inside", "active_force_magnitude"]


def analyze_one(path, out, args):
    """Analisa uma execução; só séries pequenas por célula ficam em memória."""
    out.mkdir(parents=True, exist_ok=True)
    meta = parse_metadata(path, args)
    series, prev_com, prev_whole = {}, {}, {}
    times, steps, frame_summary = [], [], []
    reference_box = None
    expected_beads = {}
    hist_edges = np.linspace(-np.pi, np.pi, 73)
    hist_abs = np.zeros(72, int)
    hist_rel = np.zeros(72, int)
    diagnostics = dict(invalid_ring_frames=0, large_com_steps=0, missing_cell_frames=0,
                       rods_outside_parent_samples=0)
    cell_file = gzip.open(out / "cells_frames.csv.gz", "wt", newline="")
    cell_writer = csv.DictWriter(cell_file, fieldnames=CELL_COLUMNS)
    cell_writer.writeheader()
    rod_file = gzip.open(out / "rods_frames.csv.gz", "wt", newline="") if args.save_rods else None
    rod_columns = ["frame", "timestep", "time", "cell_id", "rod_id", "x", "y", "ux", "uy", "theta_rad", "length", "inside_parent", "theta_minus_shape_rad"]
    rod_writer = csv.DictWriter(rod_file, fieldnames=rod_columns) if rod_file else None
    if rod_writer:
        rod_writer.writeheader()
    try:
        for frame, (step, box, c) in enumerate(read_frames(path)):
            if step < args.discard_steps:
                continue
            if reference_box is None:
                reference_box = box
            elif not (np.allclose(box.length, reference_box.length, atol=1e-7, rtol=0)
                      and np.allclose(box.lo, reference_box.lo, atol=1e-7, rtol=0)
                      and np.array_equal(box.periodic, reference_box.periodic)):
                raise ValueError("Caixa variável: analise só a produção em caixa fixa")
            if steps and step <= steps[-1]:
                raise ValueError("Timesteps devem ser estritamente crescentes")
            if len(steps) >= 2 and step-steps[-1] != steps[-1]-steps[-2]:
                raise ValueError("Intervalos de gravação variáveis: separe trechos com amostragem uniforme")
            t = step * args.dt
            dt_frame = t-times[-1] if times else np.nan
            times.append(t); steps.append(step)
            i = len(times)-1
            if len(np.unique(c["id"])) != len(c["id"]):
                raise ValueError("IDs de partículas duplicados no mesmo frame")
            rods = rod_groups(c, box, args.type_rod, args.rod_beads, args.reverse_rod_polarity)
            k = np.flatnonzero(c["type"] == args.type_ring)
            k = k[np.lexsort((c["id"][k], c["cell"][k]))]
            seen, current_rows = set(), []
            for cid, sl in groups(c["cell"][k]):
                idx = k[sl]
                seen.add(cid)
                if cid not in series:
                    if i != 0:
                        raise ValueError(f"Célula nova após o primeiro frame: {cid}")
                    series[cid] = []
                    expected_beads[cid] = c["id"][idx].copy()
                try:
                    if not np.array_equal(c["id"][idx], expected_beads[cid]):
                        raise ValueError("Contorno com partículas ausentes ou IDs alterados")
                    g = ring_geometry(c["xy"][idx], box)
                except ValueError as exc:
                    diagnostics["invalid_ring_frames"] += 1
                    if diagnostics["invalid_ring_frames"] <= 3:
                        print(f"[AVISO] {path.name}, step={step}, célula={cid}: {exc}", file=sys.stderr)
                    series[cid].append(None)
                    prev_com.pop(cid, None); prev_whole.pop(cid, None)
                    continue
                com = g.pop("com"); whole = g.pop("whole"); center = g.pop("center")
                if cid in prev_com:
                    delta = box.delta(com-prev_com[cid])
                    xu = prev_whole[cid]+delta
                    v = delta/dt_frame
                    if np.any(np.abs(delta[box.periodic]) > 0.4*box.length[box.periodic]):
                        diagnostics["large_com_steps"] += 1
                else:
                    xu = com.copy(); v = np.full(2, np.nan)
                # Após um frame inválido, inicia outra origem de imagem. MSD não cruza a lacuna.
                prev_com[cid], prev_whole[cid] = com, xu
                entries = rods.get(cid, [])
                u = np.array([r[1] for r in entries]).reshape(-1, 2)
                order = rod_order(u)
                speed = float(np.linalg.norm(v))
                vhat = v/speed if np.isfinite(speed) and speed > 1e-12 else None
                theta = np.arctan2(u[:, 1], u[:, 0]) if len(u) else np.array([])
                hist_abs += np.histogram(theta, hist_edges)[0]
                shape_angle = g["shape_theta_rad"]
                # Eixo maior da célula é apolar: theta e theta+pi são equivalentes.
                relative = (theta-shape_angle+np.pi/2) % np.pi-np.pi/2
                if np.isfinite(shape_angle):
                    hist_rel += np.histogram(relative, hist_edges)[0]
                pos = np.array([r[2] for r in entries]).reshape(-1, 2)
                pos_near = center+box.delta(pos-com) if len(pos) else pos
                inside = inside_polygon(pos_near, whole)
                diagnostics["rods_outside_parent_samples"] += int((~inside).sum())
                P = order["P"]
                cosv = float(u.mean(axis=0) @ vhat) if len(u) and vhat is not None else np.nan
                force = len(u)*meta["f_prop"] if meta["f_prop"] is not None else np.nan
                row = dict(frame=i, timestep=step, time=t, cell_id=cid, n_ring_beads=len(idx), n_rods=len(u),
                    com_x_wrapped=float(com[0]), com_y_wrapped=float(com[1]),
                    com_x_unwrapped=float(xu[0]), com_y_unwrapped=float(xu[1]),
                    velocity_x=float(v[0]), velocity_y=float(v[1]), speed_interval=speed,
                    **g, **order, rods_density_area=len(u)/g["area"] if g["area"] > 0 else np.nan,
                    rods_shape_cos2=finite_mean(np.cos(2*relative)),
                    rods_velocity_cos=cosv, polar_velocity_cos=cosv/P if np.isfinite(P) and P > 1e-10 else np.nan,
                    rods_velocity_cos2=finite_mean(2*(u @ vhat)**2-1) if vhat is not None else np.nan,
                    fraction_rods_inside=float(inside.mean()) if len(inside) else np.nan,
                    active_force_x=force*order["polar_x"], active_force_y=force*order["polar_y"],
                    active_force_magnitude=force*P)
                cell_writer.writerow(row)
                series[cid].append(row); current_rows.append(row)
                if rod_writer:
                    for j, (rid, uv, xy, length) in enumerate(entries):
                        rod_writer.writerow(dict(frame=i, timestep=step, time=t, cell_id=cid, rod_id=rid,
                            x=xy[0], y=xy[1], ux=uv[0], uy=uv[1], theta_rad=theta[j], length=length,
                            inside_parent=int(inside[j]), theta_minus_shape_rad=relative[j]))
            orphan = set(rods)-seen
            if orphan:
                raise ValueError(f"Bastonetes com célula parental sem membrana: {sorted(orphan)[:5]}")
            for cid in set(series)-seen:
                diagnostics["missing_cell_frames"] += 1
                series[cid].append(None)
                prev_com.pop(cid, None); prev_whole.pop(cid, None)
            fs = dict(frame=i, timestep=step, time=t, n_cells_valid=len(current_rows))
            for metric in METRICS:
                fs[metric+"_mean"] = finite_mean([r[metric] for r in current_rows])
            frame_summary.append(fs)
    finally:
        cell_file.close()
        if rod_file:
            rod_file.close()
    if not times or not series:
        raise ValueError("Nenhum frame/célula disponível após o descarte")
    times = np.array(times)
    summaries, msd_rows = [], []
    for cid, data in sorted(series.items()):
        valid = [r for r in data if r is not None]
        summary = dict(cell_id=cid, n_frames_valid=len(valid), n_frames_total=len(times),
                       n_ring_beads=len(expected_beads[cid]),
                       n_rods_min=min([r["n_rods"] for r in valid], default=0),
                       n_rods_max=max([r["n_rods"] for r in valid], default=0))
        for metric in METRICS:
            values = [r[metric] for r in valid]
            summary[metric+"_mean"] = finite_mean(values)
            summary[metric+"_std_time"] = finite_std(values)
        xy = np.array([[r["com_x_unwrapped"], r["com_y_unwrapped"]] if r else [np.nan, np.nan] for r in data])
        curve = msd_series(xy, times, args.msd_max_fraction)
        summary.update(fit_msd(curve, args.fit_min, args.fit_max))
        summaries.append(summary)
        for lag, tau, val, pairs in curve:
            msd_rows.append(dict(cell_id=cid, lag_frames=lag, tau=tau, msd=val, n_pairs=pairs))
    write_csv(out / "cells_summary.csv", summaries)
    write_csv(out / "frames_summary.csv", frame_summary)
    write_csv(out / "msd_cells.csv", msd_rows, ["cell_id", "lag_frames", "tau", "msd", "n_pairs"])
    msd_mean = []
    for lag in sorted(set(r["lag_frames"] for r in msd_rows)):
        rr = [r for r in msd_rows if r["lag_frames"] == lag and np.isfinite(r["msd"])]
        msd_mean.append(dict(lag_frames=lag, tau=finite_mean([r["tau"] for r in rr]),
            msd_mean=finite_mean([r["msd"] for r in rr]), msd_std_cells=finite_std([r["msd"] for r in rr]),
            n_cells=len(rr), n_pairs=sum(r["n_pairs"] for r in rr)))
    write_csv(out / "msd_mean.csv", msd_mean, ["lag_frames", "tau", "msd_mean", "msd_std_cells", "n_cells", "n_pairs"])
    hist_rows = [dict(theta_left_rad=hist_edges[j], theta_right_rad=hist_edges[j+1],
                     theta_center_rad=(hist_edges[j]+hist_edges[j+1])/2,
                     count_lab=int(hist_abs[j]), count_relative_shape=int(hist_rel[j])) for j in range(72)]
    write_csv(out / "rods_angle_histogram.csv", hist_rows)
    global_summary = dict(trajectory=str(path.resolve()), **meta, n_cells=len(series), n_frames=len(times),
        timestep_first=steps[0], timestep_last=steps[-1], dt_integrator=args.dt,
        frame_dt_median=float(np.median(np.diff(times))) if len(times) > 1 else np.nan,
        box_Lx=float(reference_box.length[0]), box_Ly=float(reference_box.length[1]),
        phi_nominal_box=len(series)*np.pi*args.radius0**2/np.prod(reference_box.length) if args.radius0 else np.nan,
        radius0=args.radius0, output_directory=str(out.resolve()), version=VERSION)
    for metric in METRICS:
        values = [r[metric+"_mean"] for r in summaries]
        global_summary[metric+"_mean"] = finite_mean(values)
        global_summary[metric+"_std_cells"] = finite_std(values)
    global_summary["alpha_eff_mean"] = finite_mean([s["alpha_eff"] for s in summaries])
    global_summary["D_slope_window_mean"] = finite_mean([s["D_slope_window"] for s in summaries])
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json_clean(dict(summary=global_summary, diagnostics=diagnostics,
            settings=vars(args), warnings=["P e S finitos não implicam ordem coletiva entre células.",
            "D_slope_window é inclinação MSD/4 na janela, não difusividade assintótica garantida."])), f, indent=2, ensure_ascii=False)
    if not args.no_plots:
        plot_run(out, summaries, frame_summary, msd_mean, hist_rows)
    return global_summary, diagnostics


def plot_run(out, cells, frames, msd, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    t = [r["time"] for r in frames]
    for ax, key, label in zip(axs.flat[:3], ["Rg_ring", "asphericity", "area"],
                             ["Cell radius of gyration", "Cell asphericity", "Cell area"]):
        ax.plot(t, [r[key+"_mean"] for r in frames]); ax.set(xlabel="Time", ylabel=label)
    axs[1, 0].plot(t, [r["P_mean"] for r in frames], label="Polar P")
    axs[1, 0].plot(t, [r["S_mean"] for r in frames], label="Nematic S")
    axs[1, 0].set(xlabel="Time", ylabel="Internal rod order", ylim=(0, 1)); axs[1, 0].legend()
    axs[1, 1].scatter([r["P_mean"] for r in cells], [r["speed_interval_mean"] for r in cells], s=15, alpha=.7)
    axs[1, 1].set(xlabel="Mean internal polarization P", ylabel="Mean interval speed")
    if msd:
        axs[1, 2].loglog([r["tau"] for r in msd], [r["msd_mean"] for r in msd])
    axs[1, 2].set(xlabel="Lag time", ylabel="Cell COM MSD")
    fig.savefig(out / "overview.png", dpi=180); fig.savefig(out / "overview.svg"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    theta = np.array([h["theta_center_rad"] for h in hist])*180/np.pi
    for ax, key, label in zip(axs, ["count_lab", "count_relative_shape"], ["Rod angle in laboratory", "Rod angle minus cell major axis"]):
        count = np.array([h[key] for h in hist], float)
        ax.bar(theta, count/count.sum() if count.sum() else count, width=5)
        ax.set(xlabel=label+" (degrees)", ylabel="Fraction per 5 degree bin")
    axs[1].set_xlim(-90, 90)
    fig.savefig(out / "rod_orientations.png", dpi=180); plt.close(fig)


def plot_campaign(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    known = [r for r in rows if r["phi"] is not None]
    forces = sorted(set(r["f_prop"] for r in known), key=lambda v: -np.inf if v is None else v)
    for force in forces:
        sub = [r for r in known if r["f_prop"] == force]
        eps_values = sorted(set(r["epsilon"] for r in sub), key=lambda v: -np.inf if v is None else v)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
        metrics = ["Rg2_ring_mean", "asphericity_mean", "speed_interval_mean", "P_mean", "S_mean", "area_mean"]
        labels = [r"Cell $R_g^2$", "Cell asphericity", "Mean interval speed", "Internal polar order P", "Internal nematic order S", "Cell area"]
        for eps in eps_values:
            points = [r for r in sub if r["epsilon"] == eps]
            phi_values = sorted(set(r["phi"] for r in points))
            for ax, metric, label in zip(axes.flat, metrics, labels):
                y = [finite_mean([r[metric] for r in points if r["phi"] == p]) for p in phi_values]
                ax.plot(phi_values, y, "o-", ms=4, label=f"eps={eps}" if eps is not None else "eps unknown")
                ax.set(xlabel="Nominal packing fraction (filename)", ylabel=label)
        axes[0, 0].legend(); fig.suptitle(f"Propulsion force = {force if force is not None else 'unknown'}")
        tag = str(force).replace(".", "p") if force is not None else "unknown"
        fig.savefig(out / f"campaign_fprop_{tag}.png", dpi=180)
        fig.savefig(out / f"campaign_fprop_{tag}.svg"); plt.close(fig)


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="Uma ou mais trajetórias .lammpstrj ou .gz")
    p.add_argument("--root", help="Pasta principal da campanha; busca recursiva por padrão")
    p.add_argument("--pattern", action="append", help="Glob de arquivos; pode repetir. Padrão: *.lammpstrj e *.lammpstrj.gz")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--out", default="analysis_cells_rods", help="Pasta de saída")
    p.add_argument("--dt", type=float, default=.001, help="Passo do integrador; tempo = TIMESTEP * dt (padrão: .001)")
    p.add_argument("--discard-steps", type=int, default=0, help="Descarta frames com TIMESTEP menor que este valor")
    p.add_argument("--radius0", type=float, help="Raio de referência para calcular phi nominal pela caixa")
    p.add_argument("--type-ring", type=int, default=0)
    p.add_argument("--type-rod", type=int, default=1)
    p.add_argument("--rod-beads", type=int, default=3)
    p.add_argument("--reverse-rod-polarity", action="store_true")
    p.add_argument("--f-prop", dest="f_prop", type=float, help="Força por bastonete; sobrepõe identificação automática")
    p.add_argument("--epsilon", type=float, help="Sobrepõe epsilon extraído do caminho")
    p.add_argument("--phi", type=float, help="Sobrepõe phi extraído do nome")
    p.add_argument("--fit-min", type=float, default=10.)
    p.add_argument("--fit-max", type=float, default=200.)
    p.add_argument("--msd-max-fraction", type=float, default=.25)
    p.add_argument("--save-rods", action="store_true", help="Salva ângulo e posição de cada bastonete em CSV.gz")
    p.add_argument("--no-plots", action="store_true", help="Dispensa Matplotlib")
    p.add_argument("--stop-on-error", action="store_true", help="Interrompe a campanha no primeiro arquivo inválido")
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.dt <= 0 or not 0 < args.msd_max_fraction <= 1 or args.fit_max <= args.fit_min or args.rod_beads < 2:
        raise SystemExit("Verifique dt, rod-beads, msd-max-fraction e janela de ajuste")
    if args.radius0 is not None and args.radius0 <= 0:
        raise SystemExit("radius0 deve ser positivo")
    if not args.no_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise SystemExit("Instale matplotlib ou use --no-plots")
    paths = [Path(p).resolve() for p in args.paths]
    if args.root or not paths:
        root = Path(args.root or ".").resolve()
        args.root = str(root)
        for pattern in args.pattern or ["*.lammpstrj", "*.lammpstrj.gz"]:
            paths.extend(root.glob(pattern) if args.no_recursive else root.rglob(pattern))
    paths = sorted(set(p.resolve() for p in paths if p.is_file()))
    if not paths:
        raise SystemExit("Nenhuma trajetória encontrada")
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for i, path in enumerate(paths, 1):
        if args.root:
            try:
                rel = path.relative_to(Path(args.root))
            except ValueError:
                rel = Path(path.name)
        else:
            rel = Path(path.name)
        stem = rel.name.removesuffix(".gz").removesuffix(".lammpstrj")
        # Hash evita colisões quando arquivos externos têm o mesmo nome.
        tag = hashlib.sha1(str(path).encode()).hexdigest()[:8]
        run_out = out / "runs" / rel.parent / (stem+"_"+tag)
        print(f"[{i}/{len(paths)}] {path}", flush=True)
        try:
            row, diagnostics = analyze_one(path, run_out, args)
            rows.append(row)
            print(f"  {row['n_cells']} células, {row['n_frames']} frames; "
                  f"Rg={row['Rg_ring_mean']:.5g}, P={row['P_mean']:.4g}, S={row['S_mean']:.4g}", flush=True)
            missing = [k for k in ("f_prop", "epsilon", "phi") if row[k] is None]
            if missing:
                print("  [AVISO] Metadados não identificados: "+", ".join(missing), flush=True)
            if any(diagnostics.values()):
                print("  [DIAGNÓSTICO] "+json.dumps(diagnostics), flush=True)
        except Exception as exc:
            failures.append(dict(trajectory=str(path), error=str(exc)))
            print("  [ERRO] "+str(exc), file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise
    if rows:
        write_csv(out / "campaign_summary.csv", rows)
        if not args.no_plots:
            plot_campaign(rows, out)
    with open(out / "campaign_manifest.json", "w", encoding="utf-8") as f:
        json.dump(json_clean(dict(version=VERSION, settings=vars(args), successes=len(rows), failures=failures)), f, indent=2, ensure_ascii=False)
    print(f"Concluído: {len(rows)} sucesso(s), {len(failures)} erro(s). Saída: {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
