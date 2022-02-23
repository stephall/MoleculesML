# models/regression.py

# Import models
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model
class LogpNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        # Initialize super class
        super().__init__()
        
        # Assign inputs to class attributes
        self.dim_in  = dim_in
        self.dim_out = dim_out
        
        # Define components
        self.fc1  = nn.Linear(dim_in, 500)
        self.fc2  = nn.Linear(500, 80)
        self.fc3  = nn.Linear(80, 10)
        self.fc4  = nn.Linear(10, dim_out)
        
    def forward(self, x):
        """ Define forward pass x->y. """
        # Propagate
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = self.fc4(x)
        
        return x