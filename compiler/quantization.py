"""Static INT8 arithmetic v2. See docs/specs/quantization.md.

This contract is deliberately versioned separately from the historical RTL.
"""
from dataclasses import dataclass
import math
import numpy as np

I32_MAX = (1 << 31) - 1


def round_away(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("non-finite numerical input")
    return np.copysign(np.floor(np.abs(value) + .5), value)


@dataclass(frozen=True)
class Quantization:
    scale: float
    zero_point: int

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if type(self.zero_point) is not int or not -128 <= self.zero_point <= 127:
            raise ValueError("INT8 zero point must be an integer in [-128,127]")

    def encode(self, value):
        return np.clip(round_away(np.asarray(value) / self.scale) + self.zero_point,
                       -128, 127).astype(np.int8)

    def decode(self, value):
        return (np.asarray(value, dtype=np.float64) - self.zero_point) * self.scale


def from_range(lo, hi):
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError("invalid calibration range")
    lo, hi = min(float(lo), 0.), max(float(hi), 0.)
    scale = (hi - lo) / 255 if hi != lo else 1.
    return Quantization(scale, int(np.clip(round_away(-128 - lo / scale), -128, 127)))


def multiplier_shift(ratio):
    """ratio ~= multiplier / 2**shift; one ties-away rounding after product."""
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("requantization ratio must be finite and positive")
    significand, exponent = math.frexp(ratio)
    multiplier = int(round_away(significand * (1 << 31)))
    if multiplier == 1 << 31:
        multiplier //= 2
        exponent += 1
    shift = 31 - exponent
    if not 0 <= shift <= 62:
        raise ValueError("requantization ratio outside v2 representable range")
    return multiplier, shift


def requantize(values, multiplier, shift, zero_point=0):
    """Python-wide products prevent host wraparound; target product is signed 64-bit."""
    multiplier, shift = int(multiplier), int(shift)
    if not 0 < multiplier <= I32_MAX or not 0 <= shift <= 62:
        raise ValueError("invalid requantization parameters")
    a = np.asarray(values)
    if a.dtype.kind not in 'iu' or np.any(a < -(1 << 31)) or np.any(a > I32_MAX):
        raise ValueError("requantizer input must fit INT32")
    product = a.astype(np.int64) * int(multiplier)
    magnitude = np.abs(product)
    if shift:
        magnitude = (magnitude + (1 << (shift - 1))) >> int(shift)
    rounded = np.where(product < 0, -magnitude, magnitude) + zero_point
    return np.clip(rounded, -128, 127).astype(np.int8)


def quantize_parameters(weight, bias, input_q, output_q):
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if w.ndim < 2 or b.shape != (w.shape[0],) or not np.isfinite(w).all():
        raise ValueError("invalid weight/bias shape or values")
    scales = np.max(np.abs(w.reshape(w.shape[0], -1)), axis=1) / 127
    scales = np.where(scales == 0, 1., scales)
    # BN-folded real models contain near-zero channels with nonzero bias. A
    # max-weight-only scale can make the product-unit bias overflow. Reserve a
    # worst-case raw/correction budget and enlarge those channel scales before
    # quantizing, without changing the float model or saturating its bias.
    reduction = int(np.prod(w.shape[1:]))
    bias_budget = I32_MAX - (128 + abs(input_q.zero_point)) * 127 * reduction - 2
    if bias_budget <= 0:
        raise ValueError('cannot reserve a safe INT32 accumulator budget')
    if not np.isfinite(b).all():
        raise ValueError('non-finite bias')
    scales = np.maximum(scales,np.abs(b)/(input_q.scale*bias_budget))
    # Maintain a representable requantizer for numerically negligible channels.
    scales = np.maximum(scales,(output_q.scale/input_q.scale)*2.**-31)
    qw = round_away(w / scales.reshape((-1,) + (1,) * (w.ndim - 1))).astype(np.int8)
    qb = round_away(b / (input_q.scale * scales))
    if np.any(qb < -(1 << 31)) or np.any(qb > I32_MAX):
        raise ValueError("bias does not fit product-scale INT32")
    qb = qb.astype(np.int64)
    flat = qw.astype(np.int64).reshape(len(qb), -1)
    corrected = qb - input_q.zero_point * flat.sum(axis=1)
    # Every raw partial sum, corrected bias, and final sum is bounded, for
    # every representable input, independent of calibration observations.
    bound = 128 * np.abs(flat).sum(axis=1) + np.abs(corrected)
    centered_bound = max(abs(-128 - input_q.zero_point), abs(127 - input_q.zero_point)) * np.abs(flat).sum(axis=1) + np.abs(qb)
    if np.any(bound > I32_MAX) or np.any(centered_bound > I32_MAX):
        raise ValueError("cannot prove all accumulator intermediates fit INT32")
    ms = [multiplier_shift(input_q.scale * s / output_q.scale) for s in scales]
    return dict(weight=qw, bias=qb.astype('<i4'), corrected_bias=corrected.astype('<i4'),
                weight_scales=scales, multiplier=np.array([m for m, _ in ms], dtype='<i4'),
                shift=np.array([s for _, s in ms], dtype=np.int8))
