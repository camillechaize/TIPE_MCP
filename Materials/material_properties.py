import numpy as np


class material_2d:
    # k: conductivity "W/(m.K)", ro: volumetric mass "kg/m3", cp: calorific capacity at constant pressure "J/(kg.K)"
    def __init__(self, k: float, ro: float, cp: float, resolution: (int, int), initial_temp_distribution_func, border_function):
        self.k = k
        self.ro = ro
        self.cp = cp
        self.alpha = k / (ro * cp)
        self.resolution = resolution
        self.initial_temp_distribution_func = initial_temp_distribution_func
        self.border_function = border_function

    def initialize_temp_array(self):
        temperature_array = np.empty(self.resolution)
        for x in range(self.resolution[0]):
            for y in range(self.resolution[1]):
                temperature_array[x, y] = self.initial_temp_distribution_func(x, y)
        return temperature_array


def constant_init_temp_func(x, y):
    return 373.15


def constant_border_temp_func(x, y, t):
    return 273.15
