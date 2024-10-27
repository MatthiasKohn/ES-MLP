import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
import os
from sklearn.metrics import f1_score

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
import timeit
import random

from torch_geometric.utils import to_dense_adj

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def generate_label_distributions(num_classes, distribution_type="Custom", seed=None):
    """
    Generates a label distribution for each label in the dataset based on the specified type.

    Parameters:
    num_classes      : int    - Number of unique labels/classes in the dataset.
    distribution_type: str    - Type of distribution to generate ("Custom" or "Uniform").
    seed             : int    - Random seed for reproducibility (optional).

    Returns:
    Dc               : dict   - Dictionary where keys are labels and values are dictionaries
                                representing the probability distribution to other labels.
    """
    if seed is not None:
        np.random.seed(seed)
    
    Dc = {}
    if distribution_type == "Custom":
        for c in range(num_classes):
            probs = np.zeros(num_classes)
            probs[(c + 1) % num_classes] = 0.5
            probs[(c + 2) % num_classes] = 0.5
            Dc[c] = {label: prob for label, prob in enumerate(probs)}
    elif distribution_type == "Uniform":
        for c in range(num_classes):
            probs = np.ones(num_classes) / num_classes  # Uniform distribution
            Dc[c] = {label: prob for label, prob in enumerate(probs)}
    else:
        raise ValueError("Invalid distribution type. Choose 'Custom' or 'Uniform'.")
    
    return Dc



def heterophile_edge_addition(data, K, distribution_type="Custom", seed=None):
    """
    Adds K heterophilous edges to the graph based on the given distribution type, only on the test mask.

    Parameters:
    data             : Data - PyTorch Geometric Data object containing the graph.
    K                : int  - Number of edges to add.
    distribution_type: str  - Type of distribution to generate ("Custom" or "Uniform").
    seed             : int  - Random seed for reproducibility (optional).

    Returns:
    data             : Data - PyTorch Geometric Data object with updated edges.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_classes = data.y.max()+1
    Dc = generate_label_distributions(num_classes, distribution_type, seed)
    print(Dc)
    # Create a mapping from labels to nodes
    label_to_nodes = {c: (data.y[data.test_mask] == c).nonzero(as_tuple=True)[0].tolist() for c in range(num_classes)}
    
    # Add K heterophilous edges
    new_edges = set()
    for _ in range(K):
        i = random.choice(data.test_mask.nonzero(as_tuple=True)[0].tolist())
        yi = data.y[i].item()
        
        # Sample a label c according to Dyi
        c = random.choices(list(Dc[yi].keys()), weights=Dc[yi].values(), k=1)[0]
        
        # Sample a node j uniformly from Vc
        j = random.choice(label_to_nodes[c])
        
        # Add edge (i, j) to the set of new edges
        new_edges.add((i, j))
    
    # Convert the set of new edges to a tensor
    new_edge_index = torch.tensor(list(new_edges), dtype=torch.long).t().contiguous()
    
    # Combine new edges with existing edges
    data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    
    return data


def getAr(edge_index, n_nodes, r):
    nnodes = edge_index.shape[1]
    
    edge_index, edge_weight = gcn_norm(edge_index, edge_weight=None, num_nodes=n_nodes)
    A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(n_nodes, n_nodes)).cuda()
    
    tmp = A
    for i in range(r-1):
        tmp = tmp.matmul(A)
        
    tmp = tmp.to_dense()
    
    return tmp

def to_one_hot(labels, num_classes):

    matrix = torch.zeros((len(labels), num_classes), dtype=torch.float)
    matrix[torch.arange(len(labels)), labels] = 1

    return matrix

def f1_scores(model, data):
    model.eval()
    
    labels = data.y
    test_mask = data.test_mask
    
    output = model(data)
    
    output = output.max(1)[1].type_as(labels)
    output = output[test_mask].cpu().detach().numpy()
    
    labels = labels[test_mask].cpu().detach().numpy()
    
    f1_micro = f1_score(output, labels, average="micro")
    f1_macro = f1_score(output, labels, average="macro")
    
    return f1_micro, f1_macro


def f1_scores_es(model, features, labels, test_mask, edge_index, num_nodes):
    
    model.eval()
    output = model(features, edge_index, num_nodes)
    
    output = output.max(1)[1].type_as(labels)
    output = output[test_mask].cpu().detach().numpy()
    
    labels = labels[test_mask].cpu().detach().numpy()
    
    f1_micro = f1_score(output, labels, average="micro")
    f1_macro = f1_score(output, labels, average="macro")
    
    return f1_micro, f1_macro



def calc_splitting_coefficients(alpha):
    A_r  = (1+alpha)/2
    A_ir = (1-alpha)/2
    
    return A_r, A_ir

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def get_r_hop_neighborhood(edge_index, num_nodes, r):
    """
    Compute the r-hop neighborhood matrix from an edge index.
    
    Parameters:
    - edge_index (torch.Tensor): The edge index of the graph (2 x num_edges).
    - num_nodes (int): The number of nodes in the graph.
    - r (int): The number of hops.
    
    Returns:
    - torch.Tensor: The r-hop neighborhood matrix (n_nodes x n_nodes).
    """
    # Convert edge index to adjacency matrix
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].cuda()
    
    # Initialize r-hop matrix as identity (0-hop neighborhood)
    r_hop_matrix = torch.eye(num_nodes).cuda()
    
    # Compute r-hop neighborhood
    current_power = adj_matrix.clone()
    for _ in range(r):
        r_hop_matrix = r_hop_matrix + current_power
        current_power = torch.matmul(current_power, adj_matrix)
    
    # Convert to binary (1 if reachable within r-hops, else 0)
    r_hop_matrix = (r_hop_matrix > 0).float()
    
    return r_hop_matrix


def mask_matrix_with_r_hop_neighborhood(matrix, edge_index, num_nodes, r):
    """
    Mask the original matrix with the r-hop neighborhood.
    
    Parameters:
    - matrix (torch.Tensor): The original matrix (n_nodes x n_nodes).
    - edge_index (torch.Tensor): The edge index of the graph (2 x num_edges).
    - num_nodes (int): The number of nodes in the graph.
    - r (int): The number of hops.
    
    Returns:
    - torch.Tensor: The masked matrix (n_nodes x n_nodes).
    """
    # Get the r-hop neighborhood matrix
    r_hop_matrix = get_r_hop_neighborhood(edge_index, num_nodes, r)
    
    # Mask the original matrix
    masked_matrix = matrix * r_hop_matrix
    
    return masked_matrix


