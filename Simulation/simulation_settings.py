class Simulation_Profile:
    def __init__(self, resolution: (int, int), time_step: float, stop_time: float):
        if resolution[0] < 2 or resolution[1] < 2:
            raise Exception("Material Resolution is too low")

        self.resolution = resolution
        self.time_step = time_step
        self.stop_time = stop_time
