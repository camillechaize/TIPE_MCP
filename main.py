from Heat_Conduction import heat_equation_solving, heat_equation_solving_GPU
from Visualizer import plot_image, video_visualize
from Visualizer.Colorize import color_heatmap
from Materials import material_properties as mp
from Simulation import simulation_settings as sp
from w_utils import console_appearance as ca


def main():
    # Create Simulation Profile
    simulation_settings = sp.Simulation_Profile((0.5, 0.5), (200, 200), 0.001, 2, (273.15, 373.15))

    # Create Material
    steel = mp.material_2d("Copper", 1.17 / 10000, 1, 1, mp.constant_init_temp_func, mp.constant_border_temp_func,
                           mp.constant_border_temp_func_gpu)

    information_on_start(steel, simulation_settings)

    # Compute Solution
    final_heat = heat_equation_solving_GPU.solve_heat_2d(steel, simulation_settings)

    # Show solution
    plot_image.visualize_heatmap_2d(final_heat[final_heat.shape[0] - 1])
    video_visualize.visualize_heatmap_2d_animation(final_heat, simulation_settings)


def information_on_start(material: mp.material_2d, simulation_profile: sp.Simulation_Profile):
    print(f'Simulation for {material.name}: \n'
          f'    PROPERTIES: \n'
          f'        alpha = {material.alpha} W/(m.K)\n'
          f'    SIMULATION Settings: \n'
          f'        geometric size = {simulation_profile.size[0]}m x {simulation_profile.size[1]}m \n'
          f'        resolution = {simulation_profile.resolution[0]}x{simulation_profile.resolution[1]} \n'
          f'        dist between 2 pixels = {(int(simulation_profile.distance_consecutive_pixels[0] * 10000)) / 10000}m X \n'
          f'                                {(int(simulation_profile.distance_consecutive_pixels[1] * 10000)) / 10000}m Y \n'
          f'        step time = {simulation_profile.time_step}s {ca.RED + ca.BOLD + ca.NEGATIVE} /!| Choose Low Step Time {ca.END}\n'
          f'        duration = {simulation_profile.stop_time}s \n'
          f'        frames number = {int(simulation_profile.stop_time // simulation_profile.time_step)}')


main()
