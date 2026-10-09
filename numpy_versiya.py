import time

import numpy as np

WIDTH = 1600
HEIGHT = 1200
MAX_ITER = 100
OUTPUT_FILE = "mandelbrot_numpy.pgm"


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


def calculate_numpy(width, height, max_iter):
    """Векторизована версія NumPy з маскою активних пікселів."""
    cx_values = -2.0 + np.arange(width, dtype=np.float64) * 2.8 / (width - 1)
    cy_values = -1.2 + np.arange(height, dtype=np.float64) * 2.4 / (height - 1)
    cx, cy = np.meshgrid(cx_values, cy_values)

    x = np.zeros((height, width), dtype=np.float64)
    y = np.zeros((height, width), dtype=np.float64)
    iterations = np.zeros((height, width), dtype=np.int32)
    active = np.ones((height, width), dtype=bool)

    for _ in range(max_iter):
        active &= (x * x + y * y <= 4.0)
        if not np.any(active):
            break

        # Оновлюємо тільки точки, які ще не вийшли за межі.
        x_old = x[active]
        y_old = y[active]
        cx_active = cx[active]
        cy_active = cy[active]

        x_new = x_old * x_old - y_old * y_old + cx_active
        y_new = 2.0 * x_old * y_old + cy_active

        x[active] = x_new
        y[active] = y_new
        iterations[active] += 1

    result = (255 * iterations / max_iter).astype(np.uint8)
    return result.ravel()


def save_pgm(filename, data, width, height):
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    with open(filename, "wb") as file:
        file.write(header)
        file.write(data.tobytes())


def main():
    print(f"Розмір: {WIDTH} x {HEIGHT}; MAX_ITER = {MAX_ITER}")

    start = time.perf_counter()
    reference = calculate_reference(WIDTH, HEIGHT, MAX_ITER)
    reference_time = time.perf_counter() - start
    print(f"Еталон Python: {reference_time:.4f} с")

    # Перший виклик тут не потребує JIT-компіляції.
    start = time.perf_counter()
    result = calculate_numpy(WIDTH, HEIGHT, MAX_ITER)
    first_time = time.perf_counter() - start

    times = []
    for run in range(1, 4):
        start = time.perf_counter()
        result = calculate_numpy(WIDTH, HEIGHT, MAX_ITER)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"NumPy, прогін {run}: {elapsed:.4f} с")

    avg = sum(times) / len(times)
    differences = int(np.count_nonzero(result != reference))

    save_pgm(OUTPUT_FILE, result, WIDTH, HEIGHT)

    print(f"\nПерший запуск NumPy: {first_time:.4f} с")
    print(f"Середній час NumPy: {avg:.4f} с")
    print(f"Прискорення відносно Python: {reference_time / avg:.2f}x")
    print(f"Розбіжних пікселів з еталоном: {differences}")
    print(f"Зображення збережено: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
