from Heat_Conduction import heat_equation_solving_GPU
from Materials import material_properties as mp
from Simulation import simulation_settings as sp
from Visualizer import video_visualize
from w_utils import informations


def main():

    # Create Simulation Profile
    simulation_settings = sp.Simulation_Profile((0.5, 0.5), (500, 500), 0.001, 10)

    # Create Material
    steel = mp.material_2d("Copper", 1.17 / 10000, 1, 1, mp.constant_init_temp_func, mp.constant_border_temp_func,
                           mp.constant_border_temp_func_gpu)

    informations.information('start', [steel, simulation_settings])

    # Compute Solution
    informations.information('start_computing')
    final_heat = heat_equation_solving_GPU.solve_heat_2d(steel, simulation_settings)

    # Show solution
    informations.information('start_video_render')
    video_visualize.visualize_heatmap_2d_animation(final_heat, simulation_settings)
    informations.information('finish')


main()
