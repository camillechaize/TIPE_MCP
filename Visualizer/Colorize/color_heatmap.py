import math
import colorsys
import numpy as np
from numba import cuda
from w_utils import project_settings as ps
from Simulation import simulation_settings as sp


def colorize_multiple_heatmaps(temperatures_array: np.ndarray, simulation_settings: sp.Simulation_Profile) -> np.ndarray:
    # Get Settings
    (min_temp, max_temp) = simulation_settings.temp_min_max
    resolution = simulation_settings.resolution
    number_iterations = temperatures_array.shape[0]
    min_temp_color_hsl, max_temp_color_hsl = rgb_to_hsl(ps.min_temp_color), rgb_to_hsl(ps.max_temp_color)

    # Grid computation
    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(resolution[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(resolution[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # All outputs
    all_outputs_heatmaps = np.empty(shape=(number_iterations, resolution[0], resolution[1], 3))

    print(min_temp, min_temp_color_hsl)
    print(max_temp, max_temp_color_hsl)

    for i in range(0, number_iterations):
        color_heatmap[blocks_per_grid, threads_per_block](temperatures_array[i], all_outputs_heatmaps[i], (min_temp, max_temp),
                                                          (min_temp_color_hsl, max_temp_color_hsl))
        all_outputs_heatmaps[i] *= 255

    return all_outputs_heatmaps


@cuda.jit()
def color_heatmap(temperature_array: np.ndarray, output_hm: np.ndarray, ex_temp: (float, float),
                  ex_temp_colors: ((float, float, float), (float, float, float))):
    # noinspection PyArgumentList
    x, y = cuda.grid(2)

    if x < temperature_array.shape[0] and y < temperature_array.shape[1]:
        # Get repartition to extreme temperatures
        t = (temperature_array[x, y] - ex_temp[0]) / (ex_temp[1] - ex_temp[0])
        # Clamp
        if t > 1:
            t = 1
        if t < 0:
            t = 0
        # "Lerp" new color
        h = (1 - t) * ex_temp_colors[0][0] + t * ex_temp_colors[1][0]
        s = (1 - t) * ex_temp_colors[0][1] + t * ex_temp_colors[1][1]
        l = (1 - t) * ex_temp_colors[0][2] + t * ex_temp_colors[1][2]
        # Convert and assign color
        output_hm[x, y] = hsl_to_rgb(h, s, l)


def rgb_to_hsl(rgb):
    r = float(rgb[0])
    g = float(rgb[1])
    b = float(rgb[2])
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, l = ((high + low) / 2,) * 3

    if high == low:
        h = 0.0
        s = 0.0
    else:
        d = high - low
        s = d / (2 - high - low) if l > 0.5 else d / (high + low)
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, l


@cuda.jit(device=True)
def hsl_to_rgb(h, s, l):
    def hue_to_rgb(p, q, t):
        t += 1 if t < 0 else 0
        t -= 1 if t > 1 else 0
        if t < 1 / 6: return p + (q - p) * 6 * t
        if t < 1 / 2: return q
        if t < 2 / 3: p + (q - p) * (2 / 3 - t) * 6
        return p

    if s == 0:
        r, g, b = l, l, l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1 / 3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1 / 3)

    return r, g, b
