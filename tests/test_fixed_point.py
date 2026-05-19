import numpy as np

from qru_registerization.fixed_point import encode_signed_magnitude, decode_signed_magnitude, quantize_signed_magnitude, max_quantization_error_bound


def test_encode_decode_known_values():
    sign, bits = encode_signed_magnitude(-0.625, 3)
    assert sign == 1
    assert bits.tolist() == [1, 0, 1]
    assert decode_signed_magnitude(sign, bits) == -0.625


def test_quantization_error_bound():
    for m in range(2, 12):
        bound = max_quantization_error_bound(m)
        for s in np.linspace(-1, 1, 2001):
            assert abs(quantize_signed_magnitude(float(s), m) - s) <= bound + 1e-12
