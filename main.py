import torch
from torch.utils.data import DataLoader

from dataset import ReceiverDataset
from mlp import MLP
from torch import nn

import numpy as np

from processing import load_data
from sklearn.model_selection import KFold

num_epochs = 500
num_folds = 5
torch.manual_seed(42)

accuracy_threshold = torch.tensor([1, 10, 0.25])

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            

def get_number_successes(outputs, targets):
    diff = torch.abs(outputs - targets)
                
    threshold_comparison_tensor = diff <= accuracy_threshold
    num_success_per_row_tensor = torch.sum(threshold_comparison_tensor, 1)
    did_row_succeed_tensor = num_success_per_row_tensor == 3
    total_successes = torch.sum(did_row_succeed_tensor)
    
    return total_successes.item()


data = load_data()

inputs = [val.get('params') for val in data]
outputs = [val.get('results') for val in data]

X = np.array(inputs)
max_X = np.max(X, axis=0)
normalized_X = X / max_X

y = np.array(outputs)

dataset = ReceiverDataset(normalized_X, y)
kfold = KFold(n_splits=num_folds, shuffle=True)

if __name__ == '__main__':
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'fold: {fold}')
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        trainloader = DataLoader(dataset, batch_size=10, sampler=train_sampler)
        testloader = DataLoader(dataset, batch_size=10, sampler=test_sampler)

        mlp = MLP(num_inputs=15, num_outputs=3)
        mlp.apply(reset_weights)
        
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(mlp.parameters(), weight_decay=1e-3)

        for epoch in range(num_epochs + 1):
            epoch_loss = 0.0
            successful_predictions = 0
            total_predictions = 0
            
            for inputs, targets in trainloader:
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 3))
                
                outputs = mlp(inputs)
                
                loss = loss_function(outputs, targets)
                total_predictions += outputs.size()[0]
                successful_predictions += get_number_successes(outputs, targets)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
            if not epoch % 100:
                print(f'epoch: {epoch}, loss: {epoch_loss:.3f}, accuracy: {successful_predictions / total_predictions}')
        print(f'training completed.')
        
        with torch.no_grad():
            successful_predictions = 0
            total_predictions = 0
            for i, (inputs, targets) in enumerate(testloader):
                
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 3))
                
                outputs = mlp(inputs)
                total_predictions += outputs.size()[0]
                successful_predictions += get_number_successes(outputs, targets)
            
            print(f'validation completed with accuracy: {successful_predictions / total_predictions}') 
        print('-------------------')
        nabers = np.array([38, 4.973684, 79.026316, 0.55163, 6, 13, 120.6923, 6.846154, 1.076923, 71.75, 200, 9.88, 31.38, 4.35, 42])
        nabers = nabers / max_X
        nabers = mlp(torch.tensor(nabers, dtype=torch.float32)) 
        reagor = [39, 3.7948717948717947, 57.64102564102564, 0.5641025641025641, 21, 12, 50.916666666666664, 3.5833333333333335, 0.4166666666666667, 70.63, 206, 9.5, 31.38, 4.47, 42.0]
        reagor = reagor / max_X
        reagor = mlp(torch.tensor(reagor, dtype=torch.float32)) 
        print(f'nabers: {nabers}')
        print(f'reagor: {reagor}')
