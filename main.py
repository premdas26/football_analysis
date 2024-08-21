import torch
from torch.utils.data import DataLoader

from dataset import ReceiverDataset
from mlp import MLP
from torch import nn

import numpy as np

from processing import load_data

if __name__ == '__main__':
    torch.manual_seed(42)

    data = load_data()

    inputs = [val.get('params') for val in data]
    outputs = [val.get('results') for val in data]

    X = np.array(inputs)
    y = np.array(outputs)

    normalized_X = (X - np.min(X)) / (np.max(X) - np.min(X))
    standardized_X = (X - np.mean(X)) / np.std(X)

    dataset = ReceiverDataset(normalized_X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    mlp = MLP(num_inputs=16, num_outputs=3)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), weight_decay=1e-3)

    for epoch in range(101):
        epoch_loss = 0.0
        for i, data in enumerate(loader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            
            targets = targets.reshape((targets.shape[0], 3))
            
            outputs = mlp(inputs)
            
            loss = loss_function(outputs, targets)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
        if not epoch % 5:
            print(f'epoch: {epoch}, loss: {loss:.3f}')
     
