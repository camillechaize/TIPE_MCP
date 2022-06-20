import numpy as np
from Materials import material_properties as mp


def solve_heat_2d(material: mp.material_2d, time_step: float, stop_time: float = 1.0) -> np.ndarray:
    temperature_array = material.initialize_temp_array()
    return temperature_array
