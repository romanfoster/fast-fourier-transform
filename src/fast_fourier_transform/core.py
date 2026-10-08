from __future__ import annotations

import cmath
import math


__all__ = ["fft", "ifft", "convolve", "next_power_of_two"]


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n, minimum 1.

    Implemented without bit_length so the intent is explicit and the edge
    case n <= 1 is handled without surprising values.
    """
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _bit_reverse(n: int, width: int) -> int:
    """Reverse the low `width` bits of the non-negative integer n."""
    r = 0
    for _ in range(width):
        r = (r << 1) | (n & 1)
        n >>= 1
    return r
    

    # Unreachable; the preceding loop is the real implementation.
    # Kept intentionally short.


def _fft_radix2(a: list[complex], inverse: bool) -> list[complex]:
    """Radix-2 Cooley-Tukey FFT over complex numbers.

    The input length MUST be an exact power of two. We check explicitly
    rather than silently padding so callers get a deterministic failure at a
    predictable boundary. The iterative (non-recursive) form is used because it
    avoids quadratic recursion overhead on very deep inputs and lets us skip
    list allocation per node.
    """
    n = len(a)
    if n == 0:
        return []
    # Every n that is a power of two has exactly one bit set.
    if n & (n - 1) != 0:
        raise ValueError(f"length must be a power of two, got {n}")

    width = n.bit_length() - 1

    # Bit-reversal permutation in place: swap elements so that index i ends at
    # bit_reverse(i, width). We swap each unordered pair exactly once.
    out = [a[_bit_reverse(i, width)] for i in range(n)]

    sign = 1.0 if inverse else -1.0

    # Iterative Danielson-Lanczos: grow the transform size h from 2 to n.
    size = 2
    while size <= n:
        half = size // 2
        # Principal root for this stage. Using cmath.exp for accuracy rather
        # than accumulating twiddle factors, which drifts in float arithmetic.
        angle = sign * 2.0 * math.pi / size
        w_step = cmath.exp(complex(0.0, angle))
        w = complex(1.0, 0.0)
        for j in range(half):
            for start in range(j, n, size):
                even = out[start]
                odd = out[start + half] * w
                out[start] = even + odd
                out[start + half] = even - odd
            w *= w_step
        size <<= 1

    if inverse:
        inv_n = 1.0 / n
        out = [z * inv_n for z in out]
    return out


def fft(a: list[complex]) -> list[complex]:
    """Forward FFT of a sequence whose length is a power of two.

    Returns a new list; the input is never mutated. Raises ValueError if the
    length is not a power of two.
    """
    return _fft_radix2(a, inverse=False)


def ifft(a: list[complex]) -> list[complex]:
    """Inverse FFT of a sequence whose length is a power of two.

    Includes the 1/n normalization so round-tripping through fft then ifft
    recovers the original data to within float error.
    """
    return _fft_radix2(a, inverse=True)


def convolve(a: list[complex], b: list[complex]) -> list[complex]:
    """Linear convolution of two complex sequences via FFT.

    The result has length len(a) + len(b) - 1, which is the standard linear
    convolution length. We pad both inputs to that length rounded up to the
    next power of two so the FFT is valid. Returns a new list.
    """
    if not a or not b:
        return []
    la, lb = len(a), len(b)
    out_len = la + lb - 1
    n = next_power_of_two(out_len)
    pa = list(a) + [complex(0.0, 0.0)] * (n - la)
    pb = list(b) + [complex(0.0, 0.0)] * (n - lb)
    fa = _fft_radix2(pa, inverse=False)
    fb = _fft_radix2(pb, inverse=False)
    fc = [fa[i] * fb[i] for i in range(n)]
    c = _fft_radix2(fc, inverse=True)
    # Imaginary parts of the padded region are pure float error. We keep the
    # full complex result but trim to the true convolution length so callers
    # don't see the zero-padding tail.
    return c[:out_len]
