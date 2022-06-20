from Heat_Conduction import heat_equation_solving
from Visualizer import plot_image
from Materials import material_properties as mp


def main():
    steel = mp.material_2d(1, 1, 1, (50, 50), mp.constant_init_temp_func, mp.constant_border_temp_func)
    time_step = 0.01
    stop_time = 6
    final_heat = heat_equation_solving.solve_heat_2d(steel, time_step, stop_time)
    plot_image.visualize_heatmap_2d(final_heat)


main()
