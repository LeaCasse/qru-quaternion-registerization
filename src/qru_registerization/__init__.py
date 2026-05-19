"""Operational tools for quaternion-guided signed registerization of single-qubit QRUs."""

from .gates import rx, ry, rz, qru_unitary, qru_state, random_qru_params
from .quaternion_geometry import su2_to_quaternion, quaternion_to_rotation_matrix, bloch_from_quaternion
from .bloch import bloch_vector, signed_coordinate
from .axis_selection import variance_axis, tangent_axis
from .amplitude_interface import projector_from_axis, projector_probability, probability_from_signed_coordinate
from .fixed_point import encode_signed_magnitude, decode_signed_magnitude, quantize_signed_magnitude

__all__ = [
    "rx", "ry", "rz", "qru_unitary", "qru_state", "random_qru_params",
    "su2_to_quaternion", "quaternion_to_rotation_matrix", "bloch_from_quaternion",
    "bloch_vector", "signed_coordinate", "variance_axis", "tangent_axis",
    "projector_from_axis", "projector_probability", "probability_from_signed_coordinate",
    "encode_signed_magnitude", "decode_signed_magnitude", "quantize_signed_magnitude",
]
