from w_utils import console_appearance as ca
from w_utils import project_settings as ps
import sys


def information(case: str, source_information=None):
    if case == 'start':
        material = source_information[0]
        simulation_profile = source_information[1]
        sys.stdout.write(f'Simulation for {material.name}: \n'
                         f'    PROPERTIES: \n'
                         f'        alpha = {material.alpha} W/(m.K)\n'
                         f'    SIMULATION Settings: \n'
                         f'        geometric size = {simulation_profile.size[0]}m x {simulation_profile.size[1]}m \n'
                         f'        resolution = {simulation_profile.resolution[0]}x{simulation_profile.resolution[1]} \n'
                         f'        dist between 2 pixels = {(int(simulation_profile.distance_consecutive_pixels[0] * 10000)) / 10000}m X \n'
                         f'                                {(int(simulation_profile.distance_consecutive_pixels[1] * 10000)) / 10000}m Y \n'
                         f'        step time = {simulation_profile.time_step}s {ca.RED + ca.BOLD + ca.NEGATIVE} /!| Choose Low Step Time {ca.END}\n'
                         f'        duration = {simulation_profile.stop_time}s \n'
                         f'        frames number simulation = {simulation_profile.simulation_frames_number} \n'
                         f'        frames number video = {simulation_profile.output_frames_number} \n'
                         f'        fps_output = {ps.frames_per_second} \n'
                         f'        ratio simulation real images = {simulation_profile.number_sim_images_for_output_image} \n')

    elif case == 'start_computing':
        sys.stdout.write(f'{ca.NEGATIVE}{ca.YELLOW} COMPUTING... {ca.END}')

    elif case == 'start_video_render':
        sys.stdout.write(f'\r{ca.NEGATIVE}{ca.CYAN} RENDERING... {ca.END}')

    elif case == 'finish':
        sys.stdout.write(f'\r{ca.NEGATIVE}{ca.GREEN} Finished {ca.END}')

    sys.stdout.flush()
