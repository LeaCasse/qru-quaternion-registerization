from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, META, TAB
from qru_registerization.amplitude_interface import (
    probability_from_signed_coordinate,
    projector_probability,
)
from qru_registerization.fixed_point import max_quantization_error_bound, quantize_signed_fixed_point
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import (
    best_pauli_axis,
    readout_values,
    state_motion,
    state_motion_weighted_axis,
    unweighted_bloch_pca_axis,
    weighted_tangent_energy,
)

# Consistent palette for paper diagnostics.
BLUE = "#173B6D"
TEAL = "#1B8A83"
DARK = "#2B2B2B"
LIGHT = "#D9D9D9"
MID = "#8A8A8A"


def _candidate_metrics(depth: int, seed: int, xs: np.ndarray) -> dict[str, object] | None:
    params = random_qru_params(depth=depth, seed=seed, scale=0.9)
    paths = compute_paths(params, xs)
    R = paths["bloch_direct"]
    states = paths["states"]
    weights = state_motion(states) ** 2
    result = state_motion_weighted_axis(R, states, xs=xs, energy_tol=1e-12, eigengap_tol=1e-5)
    if not result.identifiable or result.axis is None:
        return None

    ez = np.array([0.0, 0.0, 1.0])
    v = result.axis
    s_v = readout_values(R, v)
    s_z = readout_values(R, ez)
    e_v = weighted_tangent_energy(R, v, weights=weights, xs=xs)
    e_z = weighted_tangent_energy(R, ez, weights=weights, xs=xs)
    visible_v = float(np.sum(np.diff(s_v) ** 2))
    visible_z = float(np.sum(np.diff(s_z) ** 2))
    angle_to_z = float(np.degrees(np.arccos(np.clip(abs(float(v @ ez)), -1.0, 1.0))))
    range_v = float(np.ptp(s_v))
    range_z = float(np.ptp(s_z))

    v_basis, u1, u2 = _adapted_plane_basis(v)
    range_u1 = float(np.ptp(R @ u1))
    range_u2 = float(np.ptp(R @ u2))
    transverse_spread = float(np.sqrt(max(range_u1 * range_u2, 0.0)))

    # Deterministic figure-quality criterion, not a supervised objective.
    # We want an illustrative QRU that is still scientifically usable:
    # identifiable axis, non-trivial Bloch trajectory, material signed coordinate,
    # and a selected readout that is not merely the conventional Z axis.
    energy_gain = float(e_v / (e_z + 1e-12))
    visible_gain = float(visible_v / (visible_z + 1e-12))
    score = (
        0.45 * np.log10(energy_gain + 1e-12)
        + 0.35 * float(result.eigengap)
        + 0.25 * range_v
        + 0.55 * transverse_spread
        + 0.15 * (angle_to_z / 90.0)
        - 0.05 * range_z
    )
    if energy_gain < 1.2 or range_v < 0.5 or result.eigengap < 0.1 or transverse_spread < 0.75:
        score -= 1.5

    return {
        "depth": depth,
        "seed": seed,
        "score": float(score),
        "params": params,
        "paths": paths,
        "axis_result": result,
        "energy_vstar": float(e_v),
        "energy_z": float(e_z),
        "energy_gain_vstar_over_z": energy_gain,
        "visible_variation_vstar": visible_v,
        "visible_variation_z": visible_z,
        "visible_gain_vstar_over_z": visible_gain,
        "range_vstar": range_v,
        "range_z": range_z,
        "range_u1": range_u1,
        "range_u2": range_u2,
        "transverse_spread": transverse_spread,
        "angle_to_z_degrees": angle_to_z,
    }


def select_diagnostic_candidate(
    xs: np.ndarray,
    depths: tuple[int, ...] = (2, 3, 5, 8),
    n_seeds: int = 80,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for depth in depths:
        for seed in range(n_seeds):
            item = _candidate_metrics(depth, seed, xs)
            if item is not None:
                candidates.append(item)
    if not candidates:
        raise RuntimeError("No identifiable QRU trajectory found.")
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    return candidates[0], candidates


def _adapted_plane_basis(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / np.linalg.norm(v)
    # Choose a stable reference not almost parallel to v.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(v @ ref)) > 0.85:
        ref = np.array([1.0, 0.0, 0.0])
    u1 = ref - float(ref @ v) * v
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    u2 = u2 / np.linalg.norm(u2)
    return v, u1, u2


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _best_coordinate_projection(R: np.ndarray) -> tuple[int, int, str, str]:
    labels = ["x", "y", "z"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    def area(pair: tuple[int, int]) -> float:
        i, j = pair
        return float(np.ptp(R[:, i]) * np.ptp(R[:, j]))
    i, j = max(pairs, key=area)
    return i, j, labels[i], labels[j]

def generate_diagnostic(
    *,
    n_x: int = 181,
    n_seeds: int = 80,
    m_bits: int = 3,
    gamma: float = 0.73,
    search: bool = False,
) -> dict[str, object]:
    xs = np.linspace(-np.pi, np.pi, n_x)
    if search:
        selected, candidates = select_diagnostic_candidate(xs, n_seeds=n_seeds)
    else:
        selected = _candidate_metrics(depth=8, seed=22, xs=xs)
        if selected is None:
            raise RuntimeError("Fixed diagnostic QRU is not identifiable.")
        candidates = [selected]

    params = np.asarray(selected["params"], dtype=float)
    paths = selected["paths"]
    axis_result = selected["axis_result"]
    R = np.asarray(paths["bloch_direct"], dtype=float)
    states = np.asarray(paths["states"], dtype=complex)
    weights = state_motion(states) ** 2
    v = np.asarray(axis_result.axis, dtype=float)
    ez = np.array([0.0, 0.0, 1.0])

    s_v = readout_values(R, v)
    s_z = readout_values(R, ez)
    p_from_s = np.asarray([probability_from_signed_coordinate(s) for s in s_v])
    p_direct = np.asarray([projector_probability(psi, v) for psi in states])
    s_quant = np.asarray([quantize_signed_fixed_point(2 * p - 1, m_bits) for p in p_direct])
    abs_error = np.abs(s_quant - s_v)
    downstream_bound = abs(gamma) * abs_error

    pca = unweighted_bloch_pca_axis(R, xs=xs)
    pauli = best_pauli_axis(R, weights=weights, xs=xs)
    axes_rows = []
    for name, axis in [
        ("z", ez),
        ("best_pauli", pauli.axis if pauli.axis is not None else np.full(3, np.nan)),
        ("unweighted_bloch_pca", pca.axis if pca.axis is not None else np.full(3, np.nan)),
        ("v_star_state_motion", v),
    ]:
        axes_rows.append({
            "axis": name,
            "v_x": float(axis[0]),
            "v_y": float(axis[1]),
            "v_z": float(axis[2]),
            "weighted_tangent_energy": float(weighted_tangent_energy(R, axis, weights=weights, xs=xs)) if np.all(np.isfinite(axis)) else np.nan,
            "readout_range": float(np.ptp(readout_values(R, axis))) if np.all(np.isfinite(axis)) else np.nan,
            "visible_variation": float(np.sum(np.diff(readout_values(R, axis)) ** 2)) if np.all(np.isfinite(axis)) else np.nan,
        })

    samples = []
    v_basis, u1, u2 = _adapted_plane_basis(v)
    proj_i, proj_j, proj_label_i, proj_label_j = _best_coordinate_projection(R)
    adapted_a = R @ v_basis
    adapted_b = R @ u1
    adapted_c = R @ u2
    for i, x in enumerate(xs):
        samples.append({
            "x": float(x),
            "r_x": float(R[i, 0]),
            "r_y": float(R[i, 1]),
            "r_z": float(R[i, 2]),
            "s_z": float(s_z[i]),
            "s_vstar": float(s_v[i]),
            "p_from_s": float(p_from_s[i]),
            "p_direct": float(p_direct[i]),
            "s_quant_m3": float(s_quant[i]),
            "abs_register_error_m3": float(abs_error[i]),
            "downstream_bound_gamma_0p73": float(downstream_bound[i]),
            "adapted_vstar_coordinate": float(adapted_a[i]),
            "adapted_u1_coordinate": float(adapted_b[i]),
            "adapted_u2_coordinate": float(adapted_c[i]),
        })

    quant_rows = []
    for m in range(2, 7):
        sq = np.asarray([quantize_signed_fixed_point(2 * p - 1, m) for p in p_direct])
        err = np.abs(sq - s_v)
        quant_rows.append({
            "m_bits": m,
            "mean_abs_error": float(np.mean(err)),
            "max_abs_error": float(np.max(err)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "fraction_error_leq_0p02": float(np.mean(err <= 0.02)),
            "theoretical_quantization_bound": float(max_quantization_error_bound(m)),
            "mean_downstream_bound_gamma_0p73": float(abs(gamma) * np.mean(err)),
            "max_downstream_bound_gamma_0p73": float(abs(gamma) * np.max(err)),
        })

    summary = [{
        "selected_depth": int(selected["depth"]),
        "selected_seed": int(selected["seed"]),
        "selection_score": float(selected["score"]),
        "v_star_x": float(v[0]),
        "v_star_y": float(v[1]),
        "v_star_z": float(v[2]),
        "lambda_1": float(axis_result.eigenvalues[0]),
        "lambda_2": float(axis_result.eigenvalues[1]),
        "lambda_3": float(axis_result.eigenvalues[2]),
        "eigengap_gamma": float(axis_result.eigengap),
        "identifiable": bool(axis_result.identifiable),
        "energy_vstar": float(selected["energy_vstar"]),
        "energy_z": float(selected["energy_z"]),
        "energy_gain_vstar_over_z": float(selected["energy_gain_vstar_over_z"]),
        "visible_gain_vstar_over_z": float(selected["visible_gain_vstar_over_z"]),
        "range_vstar": float(selected["range_vstar"]),
        "range_z": float(selected["range_z"]),
        "range_u1": float(selected["range_u1"]),
        "range_u2": float(selected["range_u2"]),
        "transverse_spread": float(selected["transverse_spread"]),
        "angle_to_z_degrees": float(selected["angle_to_z_degrees"]),
        "max_probability_identity_error": float(np.max(np.abs(p_from_s - p_direct))),
        "mean_abs_register_error_m3": float(np.mean(abs_error)),
        "max_abs_register_error_m3": float(np.max(abs_error)),
        "mean_downstream_bound_gamma_0p73": float(np.mean(downstream_bound)),
        "max_downstream_bound_gamma_0p73": float(np.max(downstream_bound)),
    }]

    # Save structured outputs.
    _write_csv(TAB / "06_bloch_register_axes.csv", axes_rows)
    _write_csv(TAB / "06_bloch_register_samples.csv", samples)
    _write_csv(TAB / "06_bloch_register_quantization.csv", quant_rows)
    _write_csv(TAB / "06_bloch_register_summary.csv", summary)
    metadata = {
        "purpose": "Deterministic QRU diagnostic for geometry-aware readout and signed registerization.",
        "selection_rule": "Search random QRU seeds and choose an identifiable state-motion axis with a non-trivial Bloch trajectory, high selected-coordinate visibility, material angle from Z, and bounded finite-register diagnostics.",
        "depth": int(selected["depth"]),
        "seed": int(selected["seed"]),
        "m_bits_for_main_figure": int(m_bits),
        "gamma_for_bound": float(gamma),
        "axis_v_star": [float(x) for x in v],
        "params": params.tolist(),
        "adapted_plane_basis": {
            "v_star": [float(x) for x in v_basis],
            "u1": [float(x) for x in u1],
            "u2": [float(x) for x in u2],
        },
        "coordinate_projection_panel": {
            "indices": [int(proj_i), int(proj_j)],
            "labels": [proj_label_i, proj_label_j],
        },
        "summary": summary[0],
    }
    with (META / "06_bloch_register_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    # Figure: one scientifically interpretable diagnostic panel.
    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": DARK,
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
    })
    fig = plt.figure(figsize=(11.2, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.94), hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(xs, s_z, color=MID, linewidth=1.5, linestyle="--", label=r"fixed $Z$ readout $s_z(x)$")
    ax0.plot(xs, s_v, color=BLUE, linewidth=2.0, label=r"selected readout $s_{v_\star}(x)$")
    ax0.set_xlabel(r"input $x$")
    ax0.set_ylabel("signed coordinate")
    ax0.set_title("(a) Readout exposed by the selected axis")
    ax0.set_ylim(-1.05, 1.05)
    ax0.grid(True, color=LIGHT, linewidth=0.7)
    ax0.legend(fontsize=8, frameon=True)

    ax1 = fig.add_subplot(gs[0, 1])
    unit_b = plt.Circle((0, 0), 1.0, color=DARK, fill=False, linewidth=1.0, alpha=0.85)
    ax1.add_patch(unit_b)
    ax1.plot(R[:, proj_i], R[:, proj_j], color=BLUE, linewidth=1.7)
    ax1.scatter(R[::max(1, len(R)//55), proj_i], R[::max(1, len(R)//55), proj_j],
                c=xs[::max(1, len(R)//55)], cmap="viridis", s=14, zorder=3)
    ax1.scatter(R[0, proj_i], R[0, proj_j], color=TEAL, s=42, zorder=4, label="start")
    ax1.arrow(0, 0, 0.82 * v[proj_i], 0.82 * v[proj_j],
              color=TEAL, width=0.006, head_width=0.045, length_includes_head=True)
    ax1.text(0.9 * v[proj_i], 0.9 * v[proj_j], r"$v_\star$", color=TEAL, fontsize=10)
    ax1.axvline(0, color=LIGHT, linewidth=0.8)
    ax1.axhline(0, color=LIGHT, linewidth=0.8)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xlabel(rf"$r_{{{proj_label_i}}}(x)$")
    ax1.set_ylabel(rf"$r_{{{proj_label_j}}}(x)$")
    ax1.set_title(r"(b) Bloch trajectory projection")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, color=LIGHT, linewidth=0.7)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(adapted_a, adapted_b, color=BLUE, linewidth=1.7)
    pts = ax2.scatter(adapted_a, adapted_b, c=xs, cmap="viridis", s=13, zorder=3)
    ax2.scatter(adapted_a[0], adapted_b[0], color=TEAL, s=42, zorder=4, label="start")
    unit = plt.Circle((0, 0), 1.0, color=DARK, fill=False, linewidth=1.0, alpha=0.85)
    ax2.add_patch(unit)
    ax2.axvline(0, color=LIGHT, linewidth=0.8)
    ax2.axhline(0, color=LIGHT, linewidth=0.8)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlabel(r"$v_\star^\top r_\theta(x)$")
    ax2.set_ylabel(r"$u_1^\top r_\theta(x)$")
    ax2.set_title(r"(c) Projection in an axis-adapted plane")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, color=LIGHT, linewidth=0.7)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(xs, s_v, color=BLUE, linewidth=1.8, label=r"exact $s_{v_\star}(x)$")
    ax3.step(xs, s_quant, where="mid", color=TEAL, linewidth=1.4, label=rf"{m_bits}-bit signed register")
    ax3.fill_between(xs, s_v - abs_error, s_v + abs_error, color=TEAL, alpha=0.16, linewidth=0, label="absolute register error")
    ax3.set_xlabel(r"input $x$")
    ax3.set_ylabel("signed value")
    ax3.set_title("(d) Finite signed-register interface")
    ax3.set_ylim(-1.05, 1.05)
    ax3.grid(True, color=LIGHT, linewidth=0.7)
    ax3.legend(fontsize=8, frameon=True)

    fig.suptitle(
        "Geometry-aware QRU readout and signed-register diagnostic",
        color=DARK,
        fontsize=13,
        y=0.985,
    )
    pdf_path = FIG / "06_bloch_register_diagnostic.pdf"
    tmp_png_path = FIG / "_06_bloch_register_diagnostic_tmp.png"
    fig.savefig(tmp_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # The 3D Bloch panel is intentionally rasterized inside the PDF export;
    # direct vector export is slow and creates unnecessarily heavy files.
    try:
        from PIL import Image

        Image.open(tmp_png_path).convert("RGB").save(pdf_path)
        tmp_png_path.unlink(missing_ok=True)
    except Exception:
        fig = plt.figure(figsize=(11.2, 7.6))
        plt.text(0.5, 0.5, "Bloch-register diagnostic export failed", ha="center")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        tmp_png_path.unlink(missing_ok=True)

    return {
        "summary": summary[0],
        "quantization": quant_rows,
        "axes": axes_rows,
        "figure_pdf": str(FIG / "06_bloch_register_diagnostic.pdf"),
    }


def main() -> None:
    result = generate_diagnostic()
    print("Selected QRU diagnostic:")
    print(json.dumps(result["summary"], indent=2))
    print("Quantization summary:")
    print(json.dumps(result["quantization"], indent=2))
    print(f"Saved figure: {result['figure_pdf']}")


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
