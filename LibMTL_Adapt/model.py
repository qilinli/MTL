import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=11):
        super(Encoder, self).__init__()
        # Define three hidden layers with 256 neurons each
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        
        # Activation function
        self.activation = nn.Mish()
        
        # Dropout layer with a dropout probability of 0.1
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # First hidden layer
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
