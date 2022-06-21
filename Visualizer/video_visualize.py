import os
import numpy as np
import skvideo.io
from Simulation import simulation_settings as sp


def visualize_heatmap_2d_animation(all_heatmaps: np.ndarray, simulation_sets: sp.Simulation_Profile):
    all_heatmaps = all_heatmaps.astype(np.uint8)

    fps = all_heatmaps.shape[0] / simulation_sets.stop_time

    writer = skvideo.io.FFmpegWriter("@Results/result.avi", inputdict={'-r': str(fps)}, outputdict={'-r': str(fps)})

    for i in range(all_heatmaps.shape[0]):
        writer.writeFrame(all_heatmaps[i])

    writer.close()
