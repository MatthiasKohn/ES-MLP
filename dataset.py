import torch_geometric as tg
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.datasets import Planetoid, WikipediaNetwork, HeterophilousGraphDataset, Actor
import numpy as np
import torch
from pathlib import Path


def load_dataset(name: str, seed):
    name = name.lower()
    if name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root='data/' + name, name=name)
        data = dataset[0]
        data.train_mask, data.val_mask, data.test_mask = create_split(data, 0.6, 0.2, 0.2, seed)

    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root='data/' + name, name=name)
        data = dataset[0]
        data.train_mask, data.val_mask, data.test_mask = create_split(data, 0.6, 0.2, 0.2, seed)

    elif name in ["amazon-ratings", "roman-empire", "minesweeper"]:
        dataset = HeterophilousGraphDataset(root='data/' + name, name=name)
        data = dataset[0]
        data.train_mask, data.val_mask, data.test_mask = create_split(data, 0.6, 0.2, 0.2, seed)
    
    elif name in ["actor"]:
        dataset = Actor("data/"+ name)
        data = dataset[0]
        data.train_mask, data.val_mask, data.test_mask = create_split(data, 0.6, 0.2, 0.2, seed)

    elif name in ['dblp-easy']:
        dataset = built_dblp(name)
        data = dataset.ds
        
        data.y = data.y.squeeze()
        
        data.train_mask = data.node_year <= 2012
        data.train_mask = data.train_mask.squeeze()
        
        data.val_mask = data.node_year == 2013
        data.val_mask = data.val_mask.squeeze()
        
        data.test_mask = data.node_year > 2013
        data.test_mask = data.test_mask.squeeze()
    else:
        raise NotImplementedError(name + " Dataset is not supported")
    return data

def load_dataset2(name: str, split_number):
    path = f"data/{name}/splits/split_{split_number}.pt"
    data = torch.load(path)
    return data
    

class Dataset:
    """Class for keeping track of an item in inventory."""
    def __init__(self, ds, num_classes):
        self.ds = ds
        self.num_classes = num_classes
        
    def __getitem__(self, key):
        return self.ds

def built_dblp(name):
    path = Path('data')
    y = np.load(path/name/"y.npy")
    x = np.load(path/name/"X.npy")
    node_year = np.load(path/name/"t.npy")
    nx_graph = nx.read_adjlist(path/name/"adjlist.txt", nodetype=int)
    data = tg.utils.from_networkx(nx_graph)
    data.x = torch.tensor(x, dtype=torch.float32)
    data.y = torch.unsqueeze(torch.tensor(y), 1)
    data.node_year = torch.unsqueeze(torch.tensor(node_year),1)
    num_classes = np.unique(y).shape[0]
    ds = Dataset(data, num_classes)
    return ds

def create_split(dataset, train_portion, val_portion, test_portion, seed):
    
    y = dataset.y.cpu().detach().numpy()
    unique, counts = np.unique(y, return_counts=True)

    rng = np.random.default_rng(seed)
    train = []
    val = []
    test = []

    for cl in unique:
        
        tmp = np.argwhere(y==cl)
        c1 = int(len(tmp)*train_portion)
        c2 = int(len(tmp)*(train_portion+val_portion))
        rng.shuffle(tmp)
        train.append(tmp[:c1])
        val.append(tmp[c1:c2])
        test.append(tmp[c2:])
        
    train_ix = np.concatenate(train)
    val_ix = np.concatenate(val)
    test_ix = np.concatenate(test)

    train = torch.full_like(dataset.y, False, dtype=torch.bool)
    train[train_ix] = True
    val = torch.full_like(dataset.y, False, dtype=torch.bool)
    val[val_ix] = True
    test = torch.full_like(dataset.y, False, dtype=torch.bool)
    test[test_ix] = True
    return train, val, test


def calculate_p_q(data):
    same_class_mask = (data.y[data.edge_index[0]] == data.y[data.edge_index[1]]).float()
    diff_class_mask = 1 - same_class_mask

    num_same_class_edges = same_class_mask.sum().item()
    num_diff_class_edges = diff_class_mask.sum().item()

    p = num_same_class_edges / data.edge_index.shape[1]
    q = num_diff_class_edges / data.edge_index.shape[1]

    return p, q

