class Simulation_Profile:
    def __init__(self, size: (float, float), resolution: (int, int), time_step: float, stop_time: float, temp_min_max: (float, float)):
        if resolution[0] < 2 or resolution[1] < 2:
            raise Exception("Material Resolution is too low")

        self.size = size
        self.distance_consecutive_pixels = (size[0] / resolution[0], size[1] / resolution[1])
        self.resolution = resolution
        self.time_step = time_step
        self.stop_time = stop_time
        self.temp_min_max = temp_min_max
