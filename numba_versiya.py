import time
import cProfile
import pstats

import numpy as np
from numba import njit, prange, types

WIDTH = 1600
HEIGHT = 1200
MAX_ITER = 100
OUTPUT_FILE = "mandelbrot_numba.pgm"


def calculate_reference(width, height, max_iter):
    """Початковий послідовний алгоритм Python — еталон."""
    result = bytearray(width * height)

    for py in range(height):
        cy = -1.2 + py * 2.4 / (height - 1)

        for px in range(width):
            cx = -2.0 + px * 2.8 / (width - 1)
            x = 0.0
            y = 0.0
            iteration = 0

            while x * x + y * y <= 4.0 and iteration < max_iter:
                x_new = x * x - y * y + cx
                y = 2.0 * x * y + cy
                x = x_new
                iteration += 1

            result[py * width + px] = int(255 * iteration / max_iter)

    return np.frombuffer(result, dtype=np.uint8).copy()


@njit
def calculate_numba(width, height, max_iter):
    """JIT-компіляція Numba без паралельного виконання."""
    result = np.empty(width * height, dtype=np.uint8)

    for py in range(height):
        cy = -1.2 + py * 2.4 / (height - 1)

        for px in range(width):
            cx = -2.0 + px * 2.8 / (width - 1)
            x = 0.0
            y = 0.0
            iteration = 0

            while x * x + y * y <= 4.0 and iteration < max_iter:
                x_new = x * x - y * y + cx
                y = 2.0 * x * y + cy
                x = x_new
                iteration += 1

            result[py * width + px] = int(255 * iteration / max_iter)

    return result


@njit(parallel=True)
def calculate_numba_parallel(width, height, max_iter):
    """JIT-компіляція Numba з паралельним розподілом рядків."""
    result = np.empty(width * height, dtype=np.uint8)

    for py in prange(height):
        cy = -1.2 + py * 2.4 / (height - 1)

        for px in range(width):
            cx = -2.0 + px * 2.8 / (width - 1)
            x = 0.0
            y = 0.0
            iteration = 0

            while x * x + y * y <= 4.0 and iteration < max_iter:
                x_new = x * x - y * y + cx
                y = 2.0 * x * y + cy
                x = x_new
                iteration += 1

            result[py * width + px] = int(255 * iteration / max_iter)

    return result


def save_pgm(filename, data, width, height):
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    with open(filename, "wb") as file:
        file.write(header)
        file.write(data.tobytes())


def main():
    print(f"Розмір: {WIDTH} x {HEIGHT}; MAX_ITER = {MAX_ITER}")
    print("Створення еталона Python...")
    start = time.perf_counter()
    reference = calculate_reference(WIDTH, HEIGHT, MAX_ITER)
    reference_time = time.perf_counter() - start
    print(f"Еталон Python: {reference_time:.4f} с\n")

    signature = (types.int64, types.int64, types.int64)

    # compile() компілює функцію для вказаних типів, але не запускає її.
    print("Вимірювання компіляції Numba окремо від обчислення...")
    start = time.perf_counter()
    calculate_numba.compile(signature)
    compile_seq = time.perf_counter() - start

    start = time.perf_counter()
    calculate_numba_parallel.compile(signature)
    compile_par = time.perf_counter() - start

    print(f"Компіляція @njit:                 {compile_seq:.4f} с")
    print(f"Компіляція @njit(parallel=True): {compile_par:.4f} с\n")

    implementations = [
        ("Numba @njit", calculate_numba),
        ("Numba parallel", calculate_numba_parallel),
    ]

    results = {}
    for name, function in implementations:
        times = []
        result = None

        # Окремий перший виклик після компіляції — прогрівання.
        function(WIDTH, HEIGHT, MAX_ITER)

        for run in range(1, 4):
            start = time.perf_counter()
            result = function(WIDTH, HEIGHT, MAX_ITER)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"{name}, прогін {run}: {elapsed:.4f} с")

        differences = int(np.count_nonzero(result != reference))
        avg = sum(times) / len(times)
        results[name] = (result, avg, differences)
        print(f"{name}: середнє {avg:.4f} с; розбіжних пікселів: {differences}\n")

    save_pgm(OUTPUT_FILE, results["Numba parallel"][0], WIDTH, HEIGHT)
    print(f"Зображення збережено: {OUTPUT_FILE}")

    print("\nПідсумок:")
    print(f"Python reference: {reference_time:.4f} с")
    for name, (_, avg, differences) in results.items():
        print(f"{name}: {avg:.4f} с; прискорення {reference_time / avg:.2f}x; "
              f"розбіжних пікселів: {differences}")

    # cProfile покаже зовнішній виклик JIT-функції, а не її внутрішні цикли.
    profiler = cProfile.Profile()
    profiler.enable()
    calculate_numba(WIDTH, HEIGHT, MAX_ITER)
    profiler.disable()
    print("\nПрофіль cProfile для виклику Numba:")
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(10)


if __name__ == "__main__":
    main()
