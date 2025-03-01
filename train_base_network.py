import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import data_loader, networks, gym_utils
from utils.networks import np2torch
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    npz_file_path = 'data/gomoku_trajectories.npz'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {
        'board_size': 8,
        'history_length': 4,
        'lr': 1e-3,    
        'num_epochs': 200,
        'eval_every': 10,  
        'num_filters': 256,
        'batch_size': 32
    }
    dataset = data_loader.GomokuDataset(npz_file_path, history_length=cfg['history_length'])

    print("finished loading data")
    print("number of data points: ", len(dataset))

    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)
    
    model = networks.policyNetCNN(board_size=cfg['board_size'], input_channels=cfg['history_length'] + 3, num_filters=cfg['num_filters']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss() 

    for epoch in range(cfg['num_epochs']):
        model.train()
        total_loss = 0
        for i, (states, actions, rewards, next_states, dones) in enumerate(train_loader):
            states = states.to(device)
            actions = actions.to(device)
            import pdb; pdb.set_trace()
            optimizer.zero_grad()
            output = model(states)
            loss = criterion(output, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # if i % 100 == 0:
            #     print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')                
            #     for name, param in model.named_parameters():
            #         print(param.grad)
            #         if param.grad is not None:
            #             print(f"{name} - Gradient Mean: {param.grad.abs().mean().item()}")
        print(f"Epoch {epoch}/{cfg['num_epochs']}, Loss: {total_loss / len(train_loader):.4f}")
        if epoch % cfg['eval_every'] == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (states, actions, rewards, next_states, dones) in enumerate(val_loader):
                    states = states.to(device)
                    actions = actions.to(device)
                    predicted = model.predict(states)
                    total += actions.size(0)
                    correct += (predicted == actions).sum().item()
            print(f'Epoch {epoch}, Validation Accuracy: {100 * correct / total:.2f}%')
    torch.save(model.state_dict(), 'model.pth') 
