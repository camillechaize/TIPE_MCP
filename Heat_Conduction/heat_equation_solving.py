import numpy as np
from Materials import material_properties as mp


def solve_heat_2d(material: mp.material_2d, time_step: float, stop_time: float = 1.0) -> np.ndarray:
    temperature_array = material.initialize_temp_array()
    resolution = material.resolution
    alpha = material.alpha

    for steps in range(int(stop_time // time_step)):
        t = steps * time_step
        second_derivative_pos_array = compute_inside_second_derivative_position(temperature_array, material)
        # Inside
        for x in range(1, resolution[0] - 1):
            for y in range(1, resolution[1] - 1):
                temperature_array[x, y] = temperature_array[x, y] + time_step * second_derivative_pos_array[x, y] * alpha
        # Borders
        compute_temperature_border(temperature_array, material, t)

    return temperature_array


def compute_inside_second_derivative_position(temperature_array, material) -> np.ndarray:
    second_derivative_position_array = np.empty(material.resolution)
    for x in range(1, second_derivative_position_array.shape[0] - 1):
        for y in range(1, second_derivative_position_array.shape[1] - 1):
            second_der_pos_x = temperature_array[x + 1, y] - 2 * temperature_array[x, y] + temperature_array[x - 1, y]
            second_der_pos_y = temperature_array[x, y + 1] - 2 * temperature_array[x, y] + temperature_array[x, y - 1]
            second_derivative_position_array[x, y] = second_der_pos_x + second_der_pos_y
    return second_derivative_position_array


def compute_temperature_border(temperature_array, material, time: float):
    max_x = material.resolution[0] - 1
    max_y = material.resolution[1] - 1
    for x in range(max_x + 1):
        temperature_array[x, 0] = material.border_function(x, 0, time)
        temperature_array[x, max_y] = material.border_function(x, max_y, time)
    for y in range(max_y + 1):
        temperature_array[0, y] = material.border_function(0, y, time)
        temperature_array[max_x, y] = material.border_function(max_x, y, time)
