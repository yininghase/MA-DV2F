import os
import torch
import numpy as np

from data_process import get_problem, load_yaml
from argparse import ArgumentParser

def generate_data(config):
    
    problem_collection = np.array(config["problem collection"], dtype=int)
    assert problem_collection.shape[1] == 2 and \
        len(problem_collection.shape) == 2 and \
        np.amin(problem_collection[:,0]) >= 1 and \
        np.amin(problem_collection[:,1]) >= 0, \
        "Invalid input of problem_collection!"
    
    data_length = config["data length each case"]
    zero_velocity = config["zero velocity"]   
    collision_mode = config["collision mode"]
    parking_mode = config["parking mode"]
    
    os.makedirs(config["data folder"], exist_ok=True)
    
    for num_vehicles, num_obstacles in problem_collection:
        
        print(f"Generating Dataset {num_vehicles} Vehicles {num_obstacles} obstacles.")
        
        if config["position range"] is not None:
            position_range = config["position range"]
        else:
            position_range = max(np.sqrt(num_vehicles*250)/2, 25)
                
        vehicles, obstacles = get_problem(num_vehicles, num_obstacles, data_length, position_range, 
                                          zero_velocity, collision_mode, parking_mode)
        vehicles = np.concatenate((vehicles, np.zeros((data_length, num_vehicles, 1))), axis=-1)
        obstacles = np.concatenate((np.zeros((data_length, num_obstacles, 4)), obstacles, 
                                    np.ones((data_length, num_obstacles, 1))), axis=-1)

        data = torch.from_numpy(np.concatenate((vehicles, obstacles), axis=1))
        
        if config["is test data"]:
            data_path = os.path.join(config["data folder"], 
                                        f"test_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt")
        else:
            data_path = os.path.join(config["data folder"], 
                                        f"train_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt")
        torch.save(data, data_path)

if __name__ == "__main__":
    
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/generate_data.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    generate_data(config)
