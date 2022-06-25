import numpy as np
import skvideo.io
from Simulation import simulation_settings as sp
from Visualizer.Colorize import color_heatmap as ch


def visualize_heatmap_2d_animation(all_heatmaps: np.ndarray, simulation_sets: sp.Simulation_Profile):
    all_heatmaps_colorized = ch.colorize_multiple_heatmaps(all_heatmaps, simulation_sets)
    all_heatmaps_colorized = all_heatmaps_colorized.astype(np.uint8)

    fps = all_heatmaps_colorized.shape[0] / simulation_sets.stop_time

    writer = skvideo.io.FFmpegWriter("@Results/result.avi", inputdict={'-r': str(fps)}, outputdict={'-r': str(fps)})

    for i in range(all_heatmaps_colorized.shape[0]):
        writer.writeFrame(all_heatmaps_colorized[i])

    writer.close()
