import os
import sys
import numpy as np
from argparse import ArgumentParser

WORK_SPACE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(WORK_SPACE)

from visualization import Visualize_Velocity_Field_Single_Frame
from data_process import load_yaml, get_problem


def visualize_velocity_field_single_frame(config):
    
    num_vehicles, num_obstacles = config['problem']
    data_length = config['num examples each case']
    
    vehicles, obstacles = get_problem(num_vehicles, num_obstacles, data_length, zero_velocity=False)
    plot_name = os.path.basename(config["plot folder"])
    
    for i in range(data_length):
        # config["ego vehicle"] = vehicles[i,0,:4]
        # config["target"] = vehicles[i,0,4:]
        # config["other vehicles"] = vehicles[i,1:,:4]
        # config["obstacles"] = obstacles[i]
        # config["name"] = f"{plot_name}_vehicle={num_vehicles}_obstacle={num_obstacles}_run={i}"
        
        config["ego vehicle"] = np.array([-15.,-22., 2*np.pi/3, 1.5])
        config["target"] = np.array([-3.,10.,np.pi/2])
        config["other vehicles"] = np.array([[13.,-20., -np.pi/6, 0],[-17.,-3.,-2*np.pi/3, 0]])
        config["obstacles"] = np.array([[12.,9.,1.],[2.,-6.,2.]])
        config["name"] = f"{plot_name}_vehicle=2_obstacle=2_run={i}"
        
        visualization = Visualize_Velocity_Field_Single_Frame(config)
        visualization.plot_velocity_field()
    


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default=f"{WORK_SPACE}/configs/visualize_velocity_field.yaml", 
                        help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    problem_collection = config['problem collection']
    model_name = 'velocity_field'
    
    config["plot folder"] = os.path.join(WORK_SPACE, config["plot folder"], model_name)
    
    for i in range(len(problem_collection)):
        problem = problem_collection[i]
        print(f"current task: num_vehicle={problem[0]}, num_obstacle={problem[1]}")
        config['problem'] = problem
        visualize_velocity_field_single_frame(config)
        
    