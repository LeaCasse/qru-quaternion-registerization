from __future__ import annotations

import numpy as np


def _validate_precision(m: int) -> None:
    if not isinstance(m, (int, np.integer)) or m <= 0:
        raise ValueError("m must be a positive integer")


def encode_signed_fixed_point(s: float, m: int) -> tuple[int, np.ndarray]:
    """Encode ``s`` with one sign bit and ``m`` fractional magnitude bits.

    The representable set is
        {(-1)^b k / 2^m : b in {0,1}, k=0,...,2^m-1}.

    Inputs are clipped to [-1, 1], quantized toward zero, saturated at
    1 - 2^-m, and zero is canonicalized to sign bit 0.
    """
    _validate_precision(m)
    if not np.isfinite(s):
        raise ValueError("s must be finite")

    value = float(np.clip(s, -1.0, 1.0))
    magnitude = min(abs(value), 1.0 - 2.0 ** (-m))
    scaled = int(np.floor(magnitude * (2**m) + 1e-12))
    bits = np.array([(scaled >> (m - 1 - k)) & 1 for k in range(m)], dtype=int)
    sign_bit = int(value < 0.0 and scaled != 0)
    return sign_bit, bits


def decode_signed_fixed_point(sign_bit: int, bits: np.ndarray) -> float:
    """Decode the canonical signed-magnitude fixed-point representation."""
    if sign_bit not in (0, 1):
        raise ValueError("sign_bit must be 0 or 1")

    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size == 0:
        raise ValueError("bits must be a non-empty one-dimensional array")
    if not np.all(np.isin(bits, (0, 1))):
        raise ValueError("bits must be binary")
    bits = bits.astype(int, copy=False)

    scaled = 0
    for bit in bits:
        scaled = (scaled << 1) | int(bit)

    magnitude = scaled / (2 ** bits.size)
    if scaled == 0:
        return 0.0
    return float((-1.0 if sign_bit else 1.0) * magnitude)


def quantize_signed_fixed_point(s: float, m: int) -> float:
    sign, bits = encode_signed_fixed_point(s, m)
    return decode_signed_fixed_point(sign, bits)


def max_quantization_error_bound(m: int) -> float:
    _validate_precision(m)
    return 2.0 ** (-m)


# Backward-compatible names used by the existing pipeline.
def encode_signed_magnitude(s: float, m: int) -> tuple[int, np.ndarray]:
    return encode_signed_fixed_point(s, m)


def decode_signed_magnitude(sign_bit: int, bits: np.ndarray) -> float:
    return decode_signed_fixed_point(sign_bit, bits)


def quantize_signed_magnitude(s: float, m: int) -> float:
    return quantize_signed_fixed_point(s, m)
