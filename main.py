import torch
from torch.utils.data import DataLoader

from dataset import ReceiverDataset

import numpy as np

from processing import load_data

data = load_data()
print(data)

inputs = [val.get('params') for val in data]
outputs = [val.get('results') for val in data]

X = np.array(inputs)
y = np.array(outputs)

normalized_X = (X - min(X)) / (max(X) - min(X))
standardized_X = (X - np.mean(X)) / np.std(X)

batch_size = 64

dataset = ReceiverDataset(normalized_X, y)
loader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=1)
