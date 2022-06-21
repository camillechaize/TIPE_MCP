from Heat_Conduction import heat_equation_solving, heat_equation_solving_GPU
from Visualizer import plot_image, video_visualize
from Materials import material_properties as mp
from Simulation import simulation_settings as sp


def main():
    # Create Simulation Profile
    simulation_settings = sp.Simulation_Profile((200, 200), 0.01, 20)

    # Create Material
    steel = mp.material_2d("Steel", 10, 1, 1, mp.constant_init_temp_func, mp.constant_border_temp_func,
                           mp.constant_border_temp_func_gpu)

    information_on_start(steel, simulation_settings)

    # Compute Solution
    final_heat = heat_equation_solving_GPU.solve_heat_2d(steel, simulation_settings)

    # Show solution
    video_visualize.visualize_heatmap_2d_animation(final_heat, simulation_settings)


def information_on_start(material: mp.material_2d, simulation_profile: sp.Simulation_Profile):
    print(f'Simulation for {material.name}: \n'
          f'    PROPERTIES: \n'
          f'        alpha = {material.alpha} W/(m.K)\n'
          f'    SIMULATION Settings: \n'
          f'        resolution = {simulation_profile.resolution[0]}x{simulation_profile.resolution[1]} \n'
          f'        time step = {simulation_profile.time_step}s \n'
          f'        stop time = {simulation_profile.stop_time}s \n'
          f'        frames number = {int(simulation_profile.stop_time//simulation_profile.time_step)}')


main()
