import torch 
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, input_dim, n_hidden, n_classes, n_layers, dropout, activation):
        
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
    
        # Define first layer
        self.layers.append(nn.Linear(input_dim, n_hidden))
    
        # Hidden layers
        for i in range(n_layers-1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
    
        # Output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))

        
        

        self.layernorm = nn.LayerNorm(n_hidden, eps=1e6)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        
        self.activation = activation
    
    def forward(self, data):

        x = data.x
        for l in range(self.n_layers-1):
            x = self.layers[l](x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.activation(x)

        output = self.layers[-1](x)
        output = F.softmax(output, dim=1)
        return output