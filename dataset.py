import torch

# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
class ReceiverDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, scale_data=True):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        
    def __len__(self):
        return len(self.X)
            
    def __getitem__(self, i):
        return self.X[i], self.y[i]
            