import torch
import torch_geometric.nn as nn
import torch.nn.functional as F

class GCN(nn.models.GCN):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__(in_channels=in_feats, 
                         hidden_channels=n_hidden, 
                         out_channels=n_classes, 
                         num_layers=n_layers,
                         act=activation, 
                         dropout=dropout)

    def forward(self, data, y=None, mask=None):
        return F.softmax(super().forward(x=data.x, edge_index=data.edge_index), dim=1)