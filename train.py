import os
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt 
from tqdm import tqdm
from argparse import ArgumentParser

from gnn import IterativeGNNModel
from loss import SelfSuperVisedLearningLoss
from data_process import load_train_data, load_yaml, GNN_DataLoader, GNN_Dataset


def train_self_supervised(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device available now:', device)
    
    if not os.path.exists(config["model folder"]):
        os.makedirs(config["model folder"])
    
    model_name = config["model name"]
    # batch_split = config["batch split"]
    
    os.makedirs(os.path.join(config["model folder"], model_name), exist_ok=True)
    model_path = os.path.join(config["model folder"], model_name, f"{model_name}.pth")
    log_path = os.path.join(config["model folder"], model_name, f"{model_name}_logs.txt")
    plot_path = os.path.join(config["model folder"], model_name, f"{model_name}_learning_plot.png")
    
    if config['num samples each case'] is not None:
        lim_length = config['num samples each case']
        random_permutation = True
    else:
        lim_length = None
        random_permutation = False
        
    train_data = load_train_data(num_vehicles = config["num of vehicles"], 
                          num_obstacles = config["num of obstacles"],
                          folders = config["data folder"],
                          load_all_simpler = True,
                          lim_length = lim_length,
                          random_permutation = random_permutation,
                          )
    
    valid_data = load_train_data(num_vehicles = config["num of vehicles"], 
                          num_obstacles = config["num of obstacles"],
                          folders = config["data folder"],
                          load_all_simpler = True,
                          lim_length = lim_length,
                          random_permutation = random_permutation,
                          )

    train_dataset = GNN_Dataset(train_data, augmentation = config["augmentation"],
                                horizon = config['horizon'])
    valid_dataset = GNN_Dataset(valid_data, augmentation = config["augmentation"],
                                horizon = config['horizon'])
    
    train_data_num = len(train_dataset)
    valid_data_num = len(valid_dataset)
    print(f"Training Data: {train_data_num}")
    print(f"Validation Data: {valid_data_num}")
    
    train_loader = GNN_DataLoader(train_dataset, batch_size=config["batch size"]//config["batch split"], shuffle=True)
    valid_loader = GNN_DataLoader(valid_dataset, batch_size=config["batch size"], shuffle=False)

    model = IterativeGNNModel(horizon = config['horizon'],  
                            max_num_vehicles = config["num of vehicles"], 
                            max_num_obstacles = config["num of obstacles"],
                            mode = "self supervised training",
                            device = device,
                            )

    
    if config["pretrained model"] is not None: 
        model.load_state_dict(torch.load(config["pretrained model"]))

    model.to(device)
    print(model)

    optimizer = Adam(model.parameters(), 
                     lr = config["initial learning rate"], 
                     weight_decay = config["weight decay"])    
    scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', verbose = True, 
                                  patience = config["learning rate patience"],
                                  factor = config["learning rate factor"], 
                                  min_lr = config["min learining rate"])
    criterion = SelfSuperVisedLearningLoss(device=device)

    best_loss = np.inf

    LOGS = {
        "training_loss": [],
        "validation_loss": [],
    }

    
    f = open(log_path, 'w+')
    
    print("Start training")
    
    train_batches = []

    for epoch in tqdm(range(config["train max epochs"])):
        # === TRAIN ===
        # Sets the model in evaluation mode
        train_loss = 0
        valid_loss = 0
        cnt = 0
        model.train()

        for (inputs, batches) in train_loader:
            
            train_batches.append((inputs, batches, 0))
            
            X, B, I = zip(*train_batches)
            
            inputs = torch.cat(X, dim=0).to(device)
            batches = torch.cat(B, dim=0).to(device)
            I = list(I)
            
            for i in range(config['horizon']//config["batch split"]):
                
                optimizer.zero_grad()
                
                controls, targets, x_next_pred, edges, _ = model(inputs, batches)                

                loss = criterion(controls, targets, inputs, edges)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*len(targets)
                cnt += len(targets)
                inputs = x_next_pred.detach().clone()
                
                if config['random offset']:
                    idx_horizon = []
                    for j in range(len(I)):
                        idx_horizon += [I[j]+i]*torch.sum(B[j][:,1]).item()
                    idx_horizon = np.array(idx_horizon)
                    state_quotient = np.clip(1-idx_horizon/config['horizon'], a_max=1, a_min=0)
                    inputs = model.introduce_random_offset(inputs, state_quotient)
            
            X_, B_, I_   = [], [], []

            for i in range(len(I)):
                I[i] += config['horizon']//config["batch split"]
                if I[i] < config['horizon']:
                    X_.append(X[i])
                    B_.append(B[i])
                    I_.append(I[i])

            train_batches = list(zip(X_,B_,I_))
                
        train_loss /= cnt
        LOGS["training_loss"].append(train_loss)
        cnt = 0
        
        model.eval()
        
        with torch.no_grad():
            for (inputs, batches) in valid_loader:

                inputs = inputs.to(device)
                batches = batches.to(device)

                controls, targets, x_next_pred, edges, _ = model(inputs, batches)                
                loss = criterion(controls, targets, inputs, edges)
                valid_loss += loss.item()*len(targets)
                cnt += len(targets)
                inputs = x_next_pred.detach().clone()
                    
            valid_loss /= cnt
            LOGS["validation_loss"].append(valid_loss)

        msg = f'Epoch: {epoch+1}/{config["train max epochs"]} | Train Loss: {train_loss:.6} | Valid Loss: {valid_loss:.6}'
        with open(log_path, 'a+') as f:
            print(msg,file=f)
            print(msg)
        
        scheduler.step(valid_loss)
        
        if valid_loss < best_loss:
    
            best_loss = valid_loss
            # Reset patience (because we have improvement)
            patience_f = config["train patience"]
            torch.save(model.state_dict(), model_path)               
            print('Model Saved!')   
        
        else:
            # Decrease patience (no improvement in ROC)
            patience_f -= 1
            if patience_f == 0:
                print(f'Early stopping (no improvement since {config["train patience"]} models) | Best Valid Loss: {valid_loss:.6f}')
                break
    
    del train_dataset, train_loader, inputs, targets, batches
    
    # plot training and validation loss
    plt.plot(LOGS["training_loss"], label='Training Loss')
    plt.plot(LOGS["validation_loss"], label='Validation Loss')
    plt.text(x = 0, y = 0, s=f"best valid loss is {best_loss:6f}")
    plt.legend()
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/train.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    train_self_supervised(config)