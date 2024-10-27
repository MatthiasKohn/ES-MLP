import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn import Dropout, Linear, LayerNorm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_edge_index

class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp, self).__init__()
        self.fc_r_1  = Linear(input_dim, hid_dim)
        self.fc_ir_1 =  Linear(input_dim, hid_dim) 
        
        self.act_fn  = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_r_1.weight)
        nn.init.xavier_uniform_(self.fc_ir_1.weight)
        
        nn.init.normal_(self.fc_r_1.bias, std=1e-6)
        nn.init.normal_(self.fc_ir_1.bias, std=1e-6)
        

    def forward(self, x):
        r = self.fc_r_1(x)
        r = self.act_fn(r)
        r = self.layernorm(r)
        r = self.dropout(r)
        
        ir = self.fc_ir_1(x)
        ir = self.act_fn(ir)
        ir = self.layernorm(ir)
        ir = self.dropout(ir)
        return r, ir
    
def calc_alpha(num_nodes, lin_conv, edge_index, z_r, z_ir):
 
    
    features_cat = torch.cat((z_r[edge_index[0]], z_ir[edge_index[0]], 
                              z_r[edge_index[1]], z_ir[edge_index[1]]), dim=1)
    
    alpha_values = lin_conv(features_cat)
    
    alpha = to_dense_adj(edge_index, torch.zeros(num_nodes, dtype=torch.int64).cuda() ,edge_attr=alpha_values).squeeze()
    alpha = F.tanh(alpha)
    
    return alpha
    

def get_A_r(A, num_nodes, order):

    
    tmp = A
    
    for i in range(order-1):
        tmp = tmp.matmul(A)
        
    adj = tmp.clone().detach()
    
    edge_index, edge_weight = to_edge_index(adj.to_sparse())
    edge_index, edge_weights = gcn_norm(edge_index, edge_weight=None, num_nodes=num_nodes)

    A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weights, sparse_sizes=(num_nodes, num_nodes))

    A = A.to_dense()
    

    return A

def get_gammas(adj_r, adj_ir):
    gammas_r = adj_r/(adj_r + adj_ir)
    gammas_ir = adj_ir/(adj_r + adj_ir)
    return gammas_r, gammas_ir 
     
    
def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

class ESMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, re_eps=0.1, ir_eps=0.1):
        super(ESMLP, self).__init__()
        
        self.re_eps = re_eps
        self.ir_eps = ir_eps
        
        self.dropout = Dropout(dropout)
        
        self.nhid = nhid
        self.nfeat = nfeat
        
        self.act_fn  = torch.nn.functional.gelu
        
        self.nlayers = nlayers
        
        self.mlp = Mlp(nfeat, self.nhid, dropout)
        
        self.conv_weights = Linear(4*self.nhid,1)
        
        self.fc_r_layers  = nn.ModuleList()
        self.fc_ir_layers = nn.ModuleList()
        
        self.norm_r_layers = nn.ModuleList()
        self.norm_ir_layers = nn.ModuleList()
        
        
        
        for l in range(self.nlayers-1):
            self.fc_r_layers.append(Linear(nhid, nhid))
            self.fc_ir_layers.append(Linear(nhid, nhid))
            
            self.norm_r_layers.append(LayerNorm(nhid, eps=1e-6))
            self.norm_ir_layers.append(LayerNorm(nhid, eps=1e-6))
        
        for l in range(self.nlayers-1):
            nn.init.xavier_uniform_(self.fc_r_layers[l].weight)
            nn.init.normal_(self.fc_r_layers[l].bias, std=1e-6)
            
            nn.init.xavier_uniform_(self.fc_ir_layers[l].weight)
            nn.init.normal_(self.fc_ir_layers[l].bias, std=1e-6)
            
        
        
    
        self.classifier = Linear(self.nhid, nclass)
        

        nn.init.xavier_uniform_(self.conv_weights.weight)

    def forward(self, x, edge_index, num_nodes):
        
        
        
        #Projection into two channels
        z_0_r, z_0_ir = self.mlp(x)
        
        Z_r = z_0_r
        Z_ir = z_0_ir
        
        
        #calculate alpha weights
                            
        for l in range(self.nlayers-1):
            Z_r = self.act_fn(Z_r)
            Z_r = self.norm_r_layers[l](Z_r)
            Z_r = self.dropout(Z_r)
            Z_r = self.fc_r_layers[l](Z_r)
            
                            
            Z_ir = self.act_fn(Z_ir)
            Z_ir = self.norm_ir_layers[l](Z_ir)
            Z_ir = self.dropout(Z_ir)
            Z_ir = self.fc_ir_layers[l](Z_ir)
                            
            

        
        if self.training:
            alpha = calc_alpha(num_nodes, self.conv_weights, edge_index, Z_r, Z_ir)

    
        # Aggregation Layer Zk+1 = Z0 + Zk * W
        Z_r  = self.re_eps * z_0_r  + (1 - self.re_eps) * Z_r
        Z_ir = self.ir_eps * z_0_ir + (1 - self.ir_eps) * Z_ir
        
        if self.training:
            x_dis_r = get_feature_dis(Z_r)
            x_dis_ir = get_feature_dis(Z_ir)

        class_feature = self.classifier(Z_r)
        class_logits = F.softmax(class_feature, dim=1)

        if self.training:
            
            return class_logits, x_dis_r, x_dis_ir, alpha, Z_ir
        else:
            return class_logits
        

def Ncontrast(x_dis, adj_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss
        

def ICR_Loss(edge_index, logits, z_ir, true_label, train_mask, n_classes=-1):
    
    true_label = F.one_hot(true_label, n_classes).type(torch.float32)
    
    true_label_in = true_label[edge_index[0]]
    true_label_out = true_label[edge_index[1]]
    
    train_mask_in = train_mask[edge_index[0]]
    train_mask_out = train_mask[edge_index[1]]
    
    
    
    #edges
    y1 = logits[edge_index[0]]
    #edges
    y2 = logits[edge_index[1]]
    
    
    y1[train_mask_in] = true_label_in[train_mask_in]
    y2[train_mask_out] = true_label_out[train_mask_out]

    assert y1.shape == y2.shape
    
    
    
    label_dis = logits[edge_index[0]] * logits[edge_index[1]]
    
    label_dis = 1 - torch.sum(label_dis, dim=1)
    assert label_dis.shape[0] == edge_index.shape[1]
    
    # u denotes the source vertex of an edge
    z_u = z_ir[edge_index[0]]
    # v denotes the target vertex of an edge
    z_v = z_ir[edge_index[1]]
    
    # Calc l2 distances of the embedding of every edge
    l2_dist =  torch.pow(z_u - z_v, 2).sum(dim=1)
    
    assert l2_dist.shape[0] == edge_index.shape[1]
    # Calc the icr loss by multiplying l2 distance and label disagreement and summing them up
    loss = torch.mean(label_dis * l2_dist)
    
    # Normalize the loss
    #loss = loss/len(edge_index[0])
    
    return loss

