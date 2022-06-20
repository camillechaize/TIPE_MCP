import matplotlib.pyplot as plt
import numpy as np


def visualize_heatmap_2d(heatmap: np.ndarray):
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.show()
