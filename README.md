# fast_fourier_transform

Radix-2 Cooley-Tukey FFT over complex numbers, implemented in pure Python using only the standard library. Suitable for linear convolution and polynomial multiplication of moderately sized inputs.

## Usage

```python
from fast_fourier_transform import fft, ifft, convolve, next_power_of_two

# Forward then inverse recovers the original data (within float error).
a = [complex(1.0, 0.0), complex(2.0, 0.0), complex(3.0, 0.0), complex(4.0, 0.0)]
# ifft(fft(a)) recovers `a` to within float error; compare with a tolerance in practice.

# Convolution of two real-valued sequences, result length = len(a)+len(b)-1.
result = convolve(
    [complex(1.0, 0.0), complex(2.0, 0.0), complex(3.0, 0.0)],
    [complex(1.0, 0.0), complex(1.0, 0.0)],
)
# == [1+0j, 3+0j, 5+0j, 3+0j]

# Find the required padding length for an arbitrary input.
n = next_power_of_two(1000)  # 1024
```

`fft` and `ifft` accept and return lists of `complex`. `convolve` accepts two lists of `complex` and returns a list of `complex`. `next_power_of_two` takes and returns an `int`.

## Why this exists

The single specific problem this solves is linear convolution of two complex sequences without pulling in a heavy dependency. The implementation is the textbook iterative radix-2 Cooley-Tukey algorithm. The trade-off is simplicity and zero dependencies against performance: this is O(n log n) but will be slower than a vectorized native FFT on large inputs. It is intended for correctness-sensitive small-to-medium workloads where you cannot or do not want to add a dependency.

## Edge cases you will hit

- `fft` and `ifft` require the input length to be an exact power of two. Anything else raises `ValueError`. If your input is not a power of two, pad it yourself or use `convolve`, which pads internally to the next power of two at or above `len(a)+len(b)-1`.
- The empty sequence is a supported input to `fft` and `ifft` (returns `[]`). `convolve` returns `[]` if either operand is empty.
- Floating-point error is inherent. Round-tripping `fft` then `ifft` recovers the input to roughly 1e-9 relative tolerance; do not assert exact equality on `complex` results in your own code.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

## Limitations

Values are coerced to floats, so very large integers lose precision. If you need
exact integer aggregates over a window, this is the wrong tool.

