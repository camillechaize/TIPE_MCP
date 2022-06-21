import numpy as np
from numba import cuda


class material_2d:
    # k: conductivity "W/(m.K)", ro: volumetric mass "kg/m3", cp: calorific capacity at constant pressure "J/(kg.K)"
    def __init__(self, name: str, k: float, ro: float, cp: float, initial_temp_distribution_func,
                 border_function,
                 border_function_gpu=None):

        self.name = name
        self.k = k
        self.ro = ro
        self.cp = cp
        self.alpha = k / (ro * cp)
        self.initial_temp_distribution_func = initial_temp_distribution_func
        self.border_function = border_function
        self.border_function_gpu = border_function_gpu

    def initialize_temp_array(self, resolution: (int, int)):
        temperature_array = np.empty(resolution)
        for x in range(resolution[0]):
            for y in range(resolution[1]):
                temperature_array[x, y] = self.initial_temp_distribution_func(x, y)
        return temperature_array


def constant_init_temp_func(x, y):
    return 373.15


def constant_border_temp_func(x, y, t):
    return 273.15


@cuda.jit(device=True)
def constant_border_temp_func_gpu(x, y, t):
    return 273.15
