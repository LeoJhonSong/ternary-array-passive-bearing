import concurrent.futures
import multiprocessing

import numpy as np


class Multithreaded_Standard_Normal:
    # see: https://numpy.org/doc/stable/reference/random/multithreading.html
    def __init__(self, seed=None, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads
        self.executor = concurrent.futures.ThreadPoolExecutor(self.threads)

        seq = np.random.SeedSequence(seed)
        self._random_generators = [np.random.default_rng(s) for s in seq.spawn(threads)]

    def generate(self, size):
        n = np.prod(size)
        values = np.empty(n)
        self.step = np.ceil(n / self.threads).astype(np.int_)

        def _fill(random_state, out, first, last):
            random_state.standard_normal(out=out[first:last])

        futures = {}
        for i in range(self.threads):
            args = (_fill,
                    self._random_generators[i],
                    values,
                    i * self.step,
                    (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        return values.reshape(size).astype(np.float32)

    def __del__(self):
        self.executor.shutdown(False)


def generate_perlin_noise_2d(
        shape, res, seed: int | None = None, tileable=(False, False), interpolant=lambda t: t * t * t * (t * (t * 6 - 15) + 10)
):
    """Generate a 2D numpy array of perlin noise. #see: https://github.com/pvigier/perlin-numpy

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> noise = generate_perlin_noise_2d((256, 256), (8, 8))
    >>> plt.imshow(noise, cmap='gray', interpolation='lanczos')
    >>> plt.colorbar()
    """
    rng = np.random.default_rng(seed)
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.float32(np.pi) * rng.random((res[0] + 1, res[1] + 1), dtype=np.float32)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
