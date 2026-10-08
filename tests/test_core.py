import cmath
import math
import unittest

from fast_fourier_transform import fft, ifft, convolve, next_power_of_two


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _list_close(xs, ys, tol=1e-9):
    if len(xs) != len(ys):
        return False
    return all(_close(x, y, tol) for x, y in zip(xs, ys))


class TestNextPowerOfTwo(unittest.TestCase):
    def test_small(self):
        self.assertEqual(next_power_of_two(0), 1)
        self.assertEqual(next_power_of_two(1), 1)
        self.assertEqual(next_power_of_two(2), 2)
        self.assertEqual(next_power_of_two(3), 4)
        self.assertEqual(next_power_of_two(4), 4)
        self.assertEqual(next_power_of_two(5), 8)
        self.assertEqual(next_power_of_two(7), 8)
        self.assertEqual(next_power_of_two(8), 8)
        self.assertEqual(next_power_of_two(9), 16)

    def test_exact_power(self):
        for k in range(0, 20):
            self.assertEqual(next_power_of_two(1 << k), 1 << k)

    def test_between_powers(self):
        for k in range(0, 16):
            n = (1 << k) + 1
            self.assertEqual(next_power_of_two(n), 1 << (k + 1))


class TestFFT(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(fft([]), [])

    def test_single_element(self):
        self.assertTrue(_list_close(fft([complex(5.0, 0.0)]), [complex(5.0, 0.0)]))

    def test_dc_signal(self):
        a = [complex(1.0, 0.0)] * 8
        out = fft(a)
        self.assertTrue(_list_close(out, [complex(8.0, 0.0)] + [complex(0.0, 0.0)] * 7))

    def test_constant_offset_and_dc(self):
        a = [complex(2.0, 0.0)] * 4
        out = fft(a)
        self.assertTrue(_list_close(out, [complex(8.0, 0.0)] + [complex(0.0, 0.0)] * 3))

    def test_impulse(self):
        a = [complex(1.0, 0.0)] + [complex(0.0, 0.0)] * 7
        out = fft(a)
        # FFT of an impulse is all ones.
        self.assertTrue(_list_close(out, [complex(1.0, 0.0)] * 8))

    def test_roundtrip_power_of_two(self):
        # Deterministic seed for reproducibility.
        import random
        rng = random.Random(1234)
        a = [complex(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(32)]
        rt = ifft(fft(a))
        self.assertTrue(_list_close(rt, a, tol=1e-8))

    def test_ifft_dc(self):
        # Inverse of [8, 0, 0, ...] is the constant-1 signal.
        spec = [complex(8.0, 0.0)] + [complex(0.0, 0.0)] * 7
        out = ifft(spec)
        self.assertTrue(_list_close(out, [complex(1.0, 0.0)] * 8))

    def test_dft_matches_brute_force(self):
        def brute(a):
            n = len(a)
            return [
                sum(a[k] * cmath.exp(complex(0.0, -2.0 * math.pi * k * j / n)) for k in range(n))
                for j in range(n)
            ]
        import random
        rng = random.Random(99)
        a = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(8)]
        self.assertTrue(_list_close(fft(a), brute(a), tol=1e-9))

    def test_ifft_matches_brute_force(self):
        def brute_inv(a):
            n = len(a)
            return [
                sum(a[k] * cmath.exp(complex(0.0, 2.0 * math.pi * k * j / n)) for k in range(n)) / n
                for j in range(n)
            ]
        import random
        rng = random.Random(7)
        a = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(8)]
        self.assertTrue(_list_close(ifft(a), brute_inv(a), tol=1e-9))

    def test_invalid_length_not_power_of_two(self):
        with self.assertRaises(ValueError):
            fft([complex(1.0, 0.0)] * 3)
        with self.assertRaises(ValueError):
            ifft([complex(1.0, 0.0)] * 6)

    def test_does_not_mutate_input(self):
        a = [complex(1.0, 0.0), complex(2.0, 0.0), complex(3.0, 0.0), complex(4.0, 0.0)]
        a_copy = list(a)
        fft(a)
        self.assertEqual(a, a_copy)

    def test_complex_inputs(self):
        # Purely imaginary odd-index impulse pattern.
        a = [complex(0.0, float(i)) for i in range(4)]
        out = fft(a)
        def brute(a):
            n = len(a)
            return [
                sum(a[k] * cmath.exp(complex(0.0, -2.0 * math.pi * k * j / n)) for k in range(n))
                for j in range(n)
            ]
        self.assertTrue(_list_close(out, brute(a), tol=1e-9))


class TestConvolve(unittest.TestCase):
    def test_both_empty(self):
        self.assertEqual(convolve([], []), [])

    def test_one_empty(self):
        self.assertEqual(convolve([], [complex(1.0, 0.0)]), [])
        self.assertEqual(convolve([complex(1.0, 0.0)], []), [])

    def test_single_element_each(self):
        out = convolve([complex(3.0, 0.0)], [complex(4.0, 0.0)])
        self.assertTrue(_list_close(out, [complex(12.0, 0.0)]))

    def test_basic_real_convolution(self):
        a = [complex(1.0, 0.0), complex(2.0, 0.0), complex(3.0, 0.0)]
        b = [complex(1.0, 0.0), complex(1.0, 0.0)]
        # Direct: (1 + 2x + 3x^2)(1 + x) = 1 + 3x + 5x^2 + 3x^3
        out = convolve(a, b)
        expected = [complex(1.0, 0.0), complex(3.0, 0.0), complex(5.0, 0.0), complex(3.0, 0.0)]
        self.assertTrue(_list_close(out, expected, tol=1e-9))

    def test_length_is_linear_convolution_length(self):
        import random
        rng = random.Random(42)
        a = [complex(rng.uniform(-1, 1), 0.0) for _ in range(7)]
        b = [complex(rng.uniform(-1, 1), 0.0) for _ in range(5)]
        out = convolve(a, b)
        self.assertEqual(len(out), 7 + 5 - 1)

    def test_matches_direct_convolution(self):
        def direct(a, b):
            n = len(a) + len(b) - 1
            r = [complex(0.0, 0.0)] * n
            for i, ai in enumerate(a):
                for j, bj in enumerate(b):
                    r[i + j] += ai * bj
            return r
        import random
        rng = random.Random(31337)
        a = [complex(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(6)]
        b = [complex(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(4)]
        out = convolve(a, b)
        self.assertTrue(_list_close(out, direct(a, b), tol=1e-8))

    def test_impulse_convolution(self):
        imp = [complex(1.0, 0.0)] + [complex(0.0, 0.0)] * 7
        sig = [complex(i, 0.0) for i in range(8)]
        out = convolve(imp, sig)
        expected = sig + [complex(0.0, 0.0)] * 7
        self.assertTrue(_list_close(out, expected, tol=1e-9))

    def test_asymmetric_lengths(self):
        a = [complex(1.0, 0.0)] * 3
        b = [complex(1.0, 0.0)] * 5
        out = convolve(a, b)
        # Direct: coefficients of (1+x+x^2)(1+x+x^2+x^3+x^4)
        # = 1 + 2x + 3x^2 + 3x^3 + 3x^4 + 2x^5 + x^6
        expected = [1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0]
        self.assertTrue(_list_close(out, [complex(x, 0.0) for x in expected], tol=1e-9))


if __name__ == "__main__":
    unittest.main()
