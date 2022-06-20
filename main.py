from Heat_Conduction import heat_equation_solving
from Visualizer import plot_image
from Materials import material_properties as mp


def main():
    steel = mp.material_2d(1, 1, 1, (200, 200), mp.constant_init_temp_func, mp.constant_border_temp_func)
    time_step = 0.1
    stop_time = 2
    final_heat = heat_equation_solving.solve_heat_2d(steel, time_step, stop_time)
    plot_image.visualize_heatmap_2d(final_heat)


main()
