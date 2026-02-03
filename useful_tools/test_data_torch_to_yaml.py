import os
import glob
import torch
import yaml
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--torch_data_folder', type=str, help="torch test data folder path")
    parser.add_argument('--yaml_data_folder', type=str, help="yaml benchmark folder path")
    parser.add_argument('--map_rescale_factor', type=float, default=1, help="For GCBF or GCBF+, the map_rescale_factor should be 1.5/0.05=30")
    parser.add_argument('--flip_yaw', action='store_true', help='CL-MAPF uses left-hand system, the yaw needs to be flipped!') 
    args = parser.parse_args()
    
    torch_data_folder = args.torch_data_folder
    yaml_data_folder = args.yaml_data_folder
    map_scale_factor = args.map_scale_factor
    flip_yaw = -1 if args.flip_yaw else 1
    
    torch_data_files = glob.glob(os.path.join(torch_data_folder, f'test_data_vehicle=*_obstacle=*.pt'))
    torch_data_files.sort()
    
    for torch_file in torch_data_files:
        
        _, num_vehicles, num_obstacles = torch_file.split('=')
        num_vehicles = int(num_vehicles.split('_')[0])
        num_obstacles = int(num_obstacles.split('.')[0])
        
        print(f'Currently working on {num_vehicles} vehicles and {num_obstacles} obstacles!')
        
        torch_data = torch.load(torch_file)
        
        for i in tqdm(range(len(torch_data))):
            test_case_i = torch_data[i]
            
            vehicles = test_case_i[:num_vehicles]
            obstacles = test_case_i[num_vehicles:]
            
            vehicles[...,:2] /=  map_scale_factor
            vehicles[...,3] /=  map_scale_factor
            vehicles[...,4:6] /=  map_scale_factor
            obstacles[...,4:7] /=  map_scale_factor
            
            bias = -torch.amin(torch.cat((vehicles[:,:2], vehicles[:,4:6], obstacles[:,4:6]), dim=0), dim=0) \
                   + 25/map_scale_factor
                   
            vehicles[:,:2] += bias
            vehicles[:,4:6] += bias
            vehicles[:,2] *= flip_yaw
            vehicles[:,6] *= flip_yaw
            
            agents = [{'start': vehicles[j,:3].tolist(), 
                       'name': f'agent{j}', 
                       'goal': vehicles[j,4:7].tolist()} for j in range(len(vehicles))]
            
            if len(obstacles)>0:
                obstacles[:,4:6] += bias
                obsts = [obstacles[j, 4:6].tolist() for j in range(len(obstacles))]
            else:
                obsts = [[-1,-1]]
            
            map_size = torch.ceil(torch.amax(torch.cat((vehicles[:,:2], vehicles[:,4:6], obstacles[:,4:6]), dim=0), dim=0) 
                                  + 25/map_scale_factor).long()
            
            yaml_dict = {}
            yaml_dict['agents'] = agents
            yaml_dict['map'] = {'dimensions': map_size.tolist(), 'obstacles': obsts}
            
            if num_obstacles == 0:
                yaml_data_file = os.path.join(yaml_data_folder, 'map200by200', f'agents{num_vehicles}', 'empty',
                                             f'map_200by200_obst{num_obstacles}_agents{num_vehicles}_ex{i}.yaml')
            else:
                yaml_data_file = os.path.join(yaml_data_folder, 'map200by200', f'agents{num_vehicles}', 'obstacle',
                                             f'map_200by200_obst{num_obstacles}_agents{num_vehicles}_ex{i}.yaml')
                
            os.makedirs(os.path.join(yaml_data_folder, 'map200by200', f'agents{num_vehicles}', 'empty'), exist_ok=True)
            os.makedirs(os.path.join(yaml_data_folder, 'map200by200', f'agents{num_vehicles}', 'obstacle'), exist_ok=True)
            
            with open(yaml_data_file, 'w+') as f:
                yaml.safe_dump(yaml_dict, f)
            