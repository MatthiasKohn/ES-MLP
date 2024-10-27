import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import degree

class ESGNN_Layer(MessagePassing):
    def __init__(self, in_dim, dropout):
        super(ESGNN_Layer, self).__init__(aggr='add')  # "Add" aggregation.
        self.dropout = dropout
        self.sub_gate = nn.Linear(2 * in_dim, 1)

    def _dropout(self, input):
        return fn.dropout(input, p=self.dropout, training=self.training)

    def edge_disentangling(self, edge_index, x):
        row, col = edge_index
        z = torch.cat([x[col], x[row]], dim=1)
        
        sub_scores = torch.tanh(self.sub_gate(z))
        sub_scores = self._dropout(sub_scores)
        re_s = (1 + sub_scores) / 2
        ir_s = (1 - sub_scores) / 2

        torch.save([re_s, ir_s, re_s+ir_s], 'tensors_list.pt')
        return re_s, ir_s

    def norm_disentangling(self, edge_index, num_nodes, re_s, ir_s):
        row, col = edge_index
        deg_re = degree(col, num_nodes, dtype=torch.float)
        deg_ir = degree(col, num_nodes, dtype=torch.float)
        re_norm = deg_re.pow(-0.5)
        ir_norm = deg_ir.pow(-0.5)
        re_norm[torch.isinf(re_norm)] = 0
        ir_norm[torch.isinf(ir_norm)] = 0
        return re_norm, ir_norm

    def forward(self, x, edge_index, iter_lowpass=1):
        re_h, ir_h = x
 
        h = torch.cat((re_h, ir_h), dim=1)

        re_s, ir_s = self.edge_disentangling(edge_index, h)
        re_norm, ir_norm = self.norm_disentangling(edge_index, h.size(0), re_s, ir_s)

        for _ in range(iter_lowpass):
            re_h = self.propagate(edge_index, x=re_h, norm=re_norm, s=re_s)
            ir_h = self.propagate(edge_index, x=ir_h, norm=ir_norm, s=ir_s)
        
        return re_h, ir_h

def message(self, x_j, norm, s):
    # Ensure norm is reshaped properly
    norm = norm.view(-1, 1)  # Reshape norm to match (batch_size, 1)
    
    # Debugging prints to check shapes (remove in final version)
    print("Shapes: norm:", norm.shape, "s:", s.shape, "x_j:", x_j.shape)
    
    # Perform the message passing operation
    return norm * s * x_j


class ESGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, re_eps=0.1, ir_eps=0.1, layer_num=2, iter_lowpass=1):
        super(ESGNN, self).__init__()
        self.re_eps = re_eps
        self.ir_eps = ir_eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.iter_lowpass = iter_lowpass

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(ESGNN_Layer(hidden_dim, dropout))

        self.re_fc = nn.Linear(in_dim, hidden_dim // 2)
        self.ir_fc = nn.Linear(in_dim, hidden_dim // 2)
        self.cla = nn.Linear(hidden_dim // 2, out_dim)

    def _dropout(self, input):
        return fn.dropout(input, p=self.dropout, training=self.training)

    def forward(self, data):

        h, edge_index = data.x, data.edge_index
        
        re_h = torch.relu(self.re_fc(h))
        ir_h = torch.relu(self.ir_fc(h))
        re_h, ir_h = self._dropout(re_h), self._dropout(ir_h)
        re_raw, ir_raw = re_h, ir_h
        for layer in self.layers:
            re_h, ir_h = layer((re_h, ir_h), edge_index, self.iter_lowpass)
            re_h = self.re_eps * re_raw + re_h
            ir_h = self.ir_eps * ir_raw + ir_h
        re_z, ir_z = re_h, ir_h
        re_logits = self.cla(re_h)
        ir_logits = self.cla(ir_h)
        
        if self.training:
            return F.log_softmax(re_logits, dim=1), F.log_softmax(ir_logits, dim=1), re_z, ir_z
        else:
            return F.log_softmax(re_logits, dim=1)

class Label_Agree_Pred(nn.Module):
    def __init__(self, in_dim, dropout, metric_learnable=False):
        super(Label_Agree_Pred, self).__init__()
        if metric_learnable:
            self.pred_fc = nn.Linear(2 * in_dim, 1)
        self.dropout = dropout
        self.metric_learnable = metric_learnable

    def _dropout(self, input):
        return fn.dropout(input, p=self.dropout, training=self.training)

    def forward(self, x, edge_index):
        row, col = edge_index
        prob = torch.sigmoid(torch.sum(x[row] * x[col], dim=1))
    
        return prob

class Ir_Consistency_Loss(nn.Module):
    def __init__(self, in_dim, dropout):
        super(Ir_Consistency_Loss, self).__init__()
        self.label_agree_predictor = Label_Agree_Pred(in_dim, dropout)

    def laplacian_loss(self, edge_index, ir_h, dis_agree_e):
        row, col = edge_index
        diff_e = torch.pow(ir_h[row] - ir_h[col], 2).sum(dim=1)
        lap_loss = dis_agree_e * diff_e
        return lap_loss.mean()

    def forward(self, re_, ir_h, edge_index):
        agree_e = self.label_agree_predictor(re_, edge_index)
        dis_agree_e = 1 - agree_e
        loss = self.laplacian_loss(edge_index, ir_h, dis_agree_e)
        return loss
    

def ICR_Loss(edge_index, logits, z_ir, labels, train_mask, num_classes=-1):
    one_hot_labels = F.one_hot(labels, num_classes).type(torch.float32)
    train_labels = one_hot_labels[train_mask]

    
    
    # Create a copy of logits and modify the copy
    logits_modified = logits.clone()
    logits_modified[train_mask] = train_labels

    row, col = edge_index
    pred_labels_i = logits_modified[row]
    pred_labels_j = logits_modified[col]
    agree = (pred_labels_i * pred_labels_j).sum(dim=1, keepdim=True)
    
    disagree = 1 - agree

    z_ir_i = z_ir[row]
    z_ir_j = z_ir[col]
    loss = (disagree * (z_ir_i - z_ir_j).pow(2).sum(dim=1, keepdim=True)).mean()
    return loss

