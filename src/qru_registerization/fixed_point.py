from __future__ import annotations

import numpy as np


def encode_signed_magnitude(s: float, m: int) -> tuple[int, np.ndarray]:
    """Encode s in [-1,1] as (sign_bit, magnitude_bits).

    sign_bit = 1 for negative values, 0 otherwise.
    Magnitude uses m fractional bits and saturation at 1 - 2^-m.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    s = float(np.clip(s, -1.0, 1.0))
    sign_bit = int(s < 0)
    mag = min(abs(s), 1.0 - 2.0 ** (-m))
    scaled = int(np.floor(mag * (2**m) + 1e-12))
    bits = np.array([(scaled >> (m - 1 - k)) & 1 for k in range(m)], dtype=int)
    return sign_bit, bits


def decode_signed_magnitude(sign_bit: int, bits: np.ndarray) -> float:
    bits = np.asarray(bits, dtype=int).reshape(-1)
    m = len(bits)
    scaled = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("bits must be binary")
        scaled = (scaled << 1) | int(bit)
    mag = scaled / (2**m)
    return float((-1 if sign_bit else 1) * mag)


def quantize_signed_magnitude(s: float, m: int) -> float:
    sign, bits = encode_signed_magnitude(s, m)
    return decode_signed_magnitude(sign, bits)


def max_quantization_error_bound(m: int) -> float:
    if m <= 0:
        raise ValueError("m must be positive")
    # Floor quantization plus saturation at |s|=1 has error at most 2^-m.
    return 2.0 ** (-m)
