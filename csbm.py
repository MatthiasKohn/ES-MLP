import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np

class CSBM:
    def __init__(self, n, p, q, d=None, mu=None, sigma=None):
        self.n = n
        self.p = p
        self.q = q
        self.d = d if d else int(2*n / np.log(n*n))
        self.sigma = sigma if sigma else 0.25
        self.mu = mu if mu else 10 * self.sigma * np.sqrt(np.log(n*n) / (2 * np.sqrt(2 * self.d)))
        self.graph_data = self._generate_data()

    def _generate_features(self, n, d, classes, mu, sigma):
        features = torch.zeros(n, d, dtype=torch.float32)
        for i in range(n):
            features[i] = torch.normal((2 - classes[i]) * mu, sigma, (d,))
        return features

    def csbm(self, n, p, q, d, mu, sigma):
        G = nx.Graph()
        classes = torch.randint(2, (n,), dtype=torch.long)

        features = self._generate_features(n, d, classes, mu, sigma)

        prob_matrix = torch.rand(n, n)
        mask = (classes.unsqueeze(0) == classes.unsqueeze(1)).float()

        edge_mask = (prob_matrix < p) * mask + (prob_matrix < q) * (1 - mask)
        edge_mask = torch.triu(edge_mask, diagonal=1)

        G.add_edges_from(torch.nonzero(edge_mask, as_tuple=False).numpy())
        
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        return Data(x=features, edge_index=edge_index, y=classes)

    def _generate_data(self):
        return self.csbm(self.n, self.p, self.q, self.d, self.mu, self.sigma)


def calculate_p_q(data):
    same_class_mask = (data.y[data.edge_index[0]] == data.y[data.edge_index[1]]).float()
    diff_class_mask = 1 - same_class_mask
    num_same_class_edges = same_class_mask.sum().item()
    num_diff_class_edges = diff_class_mask.sum().item()

    p = num_same_class_edges / data.edge_index.shape[1]
    q = num_diff_class_edges / data.edge_index.shape[1]

    return p, q
