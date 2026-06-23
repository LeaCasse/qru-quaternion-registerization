from __future__ import annotations

import numpy as np

from .gates import qru_unitary, qru_state
from .quaternion_geometry import su2_to_quaternion, bloch_from_quaternion
from .bloch import bloch_vector, signed_coordinate
from .amplitude_interface import projector_probability, probability_from_signed_coordinate, signed_coordinate_from_probability
from .fixed_point import quantize_signed_fixed_point
from .quaternion_diagnostics import canonical_orient_axis


def compute_paths(params: np.ndarray, xs: np.ndarray) -> dict[str, np.ndarray]:
    qs, rb, rq, states, unitaries = [], [], [], [], []
    for x in xs:
        U = qru_unitary(params, float(x))
        state = qru_state(params, float(x))
        q = su2_to_quaternion(U)
        unitaries.append(U)
        qs.append(q)
        rb.append(bloch_vector(state))
        rq.append(bloch_from_quaternion(q))
        states.append(state)
    return {
        "xs": np.asarray(xs, dtype=float),
        "quaternions": np.asarray(qs, dtype=float),
        "bloch_direct": np.asarray(rb, dtype=float),
        "bloch_quaternion": np.asarray(rq, dtype=float),
        "states": np.asarray(states, dtype=complex),
        "unitaries": np.asarray(unitaries, dtype=complex),
    }


def registerize_path(states: np.ndarray, bloch_path: np.ndarray, v: np.ndarray, m: int) -> dict[str, np.ndarray]:
    axis = canonical_orient_axis(v)
    s = np.array([signed_coordinate(r, axis) for r in bloch_path], dtype=float)
    p_from_s = np.array([probability_from_signed_coordinate(si) for si in s], dtype=float)
    p_direct = np.array([projector_probability(psi, axis) for psi in states], dtype=float)
    s_from_p = np.array([signed_coordinate_from_probability(pi) for pi in p_direct], dtype=float)
    s_quant = np.array([quantize_signed_fixed_point(si, m) for si in s_from_p], dtype=float)
    return {"axis": axis, "s": s, "p_from_s": p_from_s, "p_direct": p_direct, "s_from_p": s_from_p, "s_quant": s_quant}
