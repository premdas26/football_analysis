from torch import nn

class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 264),
            nn.ReLU(),
            nn.Linear(264, 264),
            nn.ReLU(),
            nn.Linear(264, num_outputs)
        )
    
    def forward(self, X):
        return self.layers(X)