import math
import numpy as np
from numba import cuda
from Materials import material_properties as mp
from Simulation import simulation_settings as sp

border_function_gpu = None


def solve_heat_2d(material: mp.material_2d, simulation_settings: sp.Simulation_Profile) -> np.ndarray:
    # Get Settings
    resolution = simulation_settings.resolution
    time_step = simulation_settings.time_step
    stop_time = simulation_settings.stop_time
    alpha = material.alpha
    global border_function_gpu
    border_function_gpu = material.border_function_gpu
    number_iterations = simulation_settings.simulation_frames_number

    # Grid computation
    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(resolution[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(resolution[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # All outputs
    all_outputs_temperatures = np.empty(shape=(simulation_settings.output_frames_number, resolution[0], resolution[1]))
    temporary_last_temperature = material.initialize_temp_array(resolution)
    temporary_next_temperature = np.empty(shape=(resolution[0], resolution[1]))

    for steps in range(0, number_iterations - 1):
        t = steps * time_step
        compute_inside_second_derivative_position[blocks_per_grid, threads_per_block](temporary_last_temperature,
                                                                                      temporary_next_temperature, alpha,
                                                                                      simulation_settings.distance_consecutive_pixels, t,
                                                                                      time_step)

        all_outputs_temperatures[(steps // simulation_settings.number_sim_images_for_output_image)] = temporary_next_temperature

        temporary_last_temperature = temporary_next_temperature

    return all_outputs_temperatures


@cuda.jit()
def compute_inside_second_derivative_position(temperature_array: np.ndarray, output_temperature_array: np.ndarray, material_alpha: float,
                                              distance_pixel: (float, float),
                                              t: float,
                                              time_step: float):
    # noinspection PyArgumentList
    x, y = cuda.grid(2)
    if 0 < x < temperature_array.shape[0] - 1 and 0 < y < temperature_array.shape[1] - 1:
        second_der_pos_x = (temperature_array[x + 1, y] - 2 * temperature_array[x, y] + temperature_array[x - 1, y]) / (distance_pixel[
                                                                                                                            0] ** 2)
        second_der_pos_y = (temperature_array[x, y + 1] - 2 * temperature_array[x, y] + temperature_array[x, y - 1]) / (distance_pixel[
                                                                                                                            1] ** 2)
        output_temperature_array[x, y] = time_step * (second_der_pos_x + second_der_pos_y) * material_alpha + temperature_array[x, y]
    elif x == 0 or x == temperature_array.shape[0] - 1 or y == 0 or y == temperature_array.shape[1] - 1:
        # noinspection PyCallingNonCallable
        output_temperature_array[x, y] = border_function_gpu(x, y, t)


def compute_temperature_border(temperature_array, material, time: float):
    max_x = material.resolution[0] - 1
    max_y = material.resolution[1] - 1
    for x in range(max_x + 1):
        temperature_array[x, 0] = material.border_function(x, 0, time)
        temperature_array[x, max_y] = material.border_function(x, max_y, time)
    for y in range(max_y + 1):
        temperature_array[0, y] = material.border_function(0, y, time)
        temperature_array[max_x, y] = material.border_function(max_x, y, time)
