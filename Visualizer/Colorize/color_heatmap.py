import math
import numpy as np
from numba import cuda
from w_utils import project_settings as ps
from Simulation import simulation_settings as sp


def colorize_multiple_heatmaps(temperatures_array: np.ndarray, simulation_settings: sp.Simulation_Profile) -> np.ndarray:
    # Get Settings
    resolution = simulation_settings.resolution
    number_iterations = temperatures_array.shape[0]
    ex_temp_color = np.array([rgb_to_hsl(ps.ex_temp_color[n]) for n in range(len(ps.ex_temp_color))])

    # Grid computation
    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(resolution[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(resolution[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # All outputs
    all_outputs_heatmaps = np.empty(shape=(number_iterations, resolution[0], resolution[1], 3))

    for i in range(0, number_iterations):
        color_heatmap[blocks_per_grid, threads_per_block](temperatures_array[i], all_outputs_heatmaps[i], ps.ex_temp,
                                                          ex_temp_color)
        all_outputs_heatmaps[i] *= 255

    return all_outputs_heatmaps


@cuda.jit()
def color_heatmap(temperature_array: np.ndarray, output_hm: np.ndarray, ex_temp, ex_temp_colors):
    # noinspection PyArgumentList
    x, y = cuda.grid(2)

    if x < temperature_array.shape[0] and y < temperature_array.shape[1]:
        # Calculate the space between two colored temperatures relative to the other ones
        num_ex_temp = len(ex_temp)
        for i in range(num_ex_temp - 1):
            # Get repartition to extreme temperatures
            t = (temperature_array[x, y] - ex_temp[i]) / (ex_temp[i + 1] - ex_temp[i])

            # Check border cases: temperature of pixel is outside the user-defined range
            if i == 0 and t < 0:
                t = 0
            elif i == num_ex_temp - 2 and t > 1:
                t = 1

            # If temperature is inside space: color pixel
            if 0 <= t <= 1:
                # "Lerp" new color
                h = (1 - t) * ex_temp_colors[i][0] + t * ex_temp_colors[i + 1][0]
                s = (1 - t) * ex_temp_colors[i][1] + t * ex_temp_colors[i + 1][1]
                lum = (1 - t) * ex_temp_colors[i][2] + t * ex_temp_colors[i + 1][2]
                # Convert and assign color
                output_hm[x, y] = hsl_to_rgb(h, s, lum)
                break


def rgb_to_hsl(rgb):
    r = float(rgb[0])
    g = float(rgb[1])
    b = float(rgb[2])
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, lum = ((high + low) / 2,) * 3

    if high == low:
        h = 0.0
        s = 0.0
    else:
        d = high - low
        s = d / (2 - high - low) if lum > 0.5 else d / (high + low)
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, lum


@cuda.jit(device=True)
def hsl_to_rgb(h, s, lum):
    def hue_to_rgb(p_1, q_1, t):
        t += 1 if t < 0 else 0
        t -= 1 if t > 1 else 0
        if t < 1 / 6:
            return p_1 + (q_1 - p_1) * 6 * t
        if t < 1 / 2:
            return q_1
        if t < 2 / 3:
            p_1 + (q_1 - p_1) * (2 / 3 - t) * 6
        return p_1

    if s == 0:
        r, g, b = lum, lum, lum
    else:
        q = lum * (1 + s) if lum < 0.5 else lum + s - lum * s
        p = 2 * lum - q
        r = hue_to_rgb(p, q, h + 1 / 3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1 / 3)

    return r, g, b
