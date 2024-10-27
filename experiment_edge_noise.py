from parse import args
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric as tg
from tqdm import tqdm
import numpy as np

import pandas as pd
from torchmetrics import AUROC

from result_writer import write_score_es, create_setting_folder_es, create_csbm_folder, write_csbm_score, create_setting_folder_edge_noise, write_score_edge_noise

from utils import accuracy, f1_scores, getAr, calc_splitting_coefficients
from utils import get_r_hop_neighborhood, mask_matrix_with_r_hop_neighborhood, heterophile_edge_addition
from dataset import load_dataset, load_dataset2
from models.esgnn import ESGNN,Ir_Consistency_Loss

from models.mlp import MLP
from models.graph_mlp_original import GMLP, Ncontrast
from models.gcn import GCN
from models.graph_sage import GraphSage
from models.es_mlp import ESMLP, get_gammas, ICR_Loss, get_order_adj, Ncontrast
from models.esgnn import ESGNN, Ir_Consistency_Loss
from models.linkx import LINKX
from torch_geometric.utils import homophily, is_undirected
from pathlib import Path
from early_stopping import EarlyStopping

import timeit
import random
from pathlib import Path

import logging

logging.basicConfig(filename="logging.log", level=logging.INFO)
logger = logging.getLogger()


def train_es_mlp(epoch, args, model, data, labels, adj_mask):

    num_nodes = data.x.size(0)
    
    output, x_dis_r, x_dis_ir, alpha, Z_ir = model(data)

    A_r, A_ir = calc_splitting_coefficients(alpha)

    A_r = get_order_adj(A_r, num_nodes, args.order)
    A_ir = get_order_adj(A_ir, num_nodes, args.order)


    gamma_r, gamma_ir = get_gammas(A_r, A_ir, adj_mask) 

    # Calculate Cross entropy loss
    loss_train_class = F.cross_entropy(output[data.train_mask], labels[data.train_mask])

    # Calculate NContrast Loss
    loss_NcontrastR = Ncontrast(x_dis_r, gamma_r, args.tau)    
    loss_NcontrastIR = Ncontrast(x_dis_ir, gamma_ir, args.tau)  
    loss_Ncon = (loss_NcontrastR + loss_NcontrastIR)/2  

    loss_icr = ICR_Loss(data.edge_index, output, Z_ir, labels, data.train_mask)

    # Combine all loses 
    loss =  loss_train_class + loss_Ncon * args.alpha + loss_icr * args.beta
    
    return loss


def train_esgnn(model, args, data, labels, ir_loss_module):

    re_logits, _, _, ir_z = model(data)

    loss_train_class = F.nll_loss(re_logits[data.train_mask], labels[data.train_mask])

    ir_loss = ir_loss_module(re_logits, ir_z, data.edge_index)

    loss = loss_train_class + args.beta * ir_loss

    return loss

    
def train_basic_model(model, data, labels):
    output = model(data)
    if model == "linkx":
        loss =  F.nll_loss(output[data.train_mask], labels[data.train_mask])
    else:
        loss = F.cross_entropy(output[data.train_mask], labels[data.train_mask])
    # print(loss)
    return loss

def train_graph_mlp(args, model, data, labels, adj_label):
    output, x_dis = model(data)
    loss_train_class = F.nll_loss(output[data.train_mask], labels[data.train_mask])
    loss_Ncontrast = Ncontrast(x_dis, adj_label, tau=args.tau)
    loss = loss_train_class + loss_Ncontrast * args.alpha
    return loss

def get_batch(data, batch_size, n_nodes, adj_label):
        """
        get a batch of feature & adjacency matrix
        """
        indices = torch.randperm(n_nodes)[:batch_size].cuda()
        data_batch = data.subgraph(indices)
        if adj_label != None:
            adj_label_batch = adj_label[indices][:, indices]
            return data_batch, adj_label_batch
        else:
            return data_batch






def train(epoch, args, model, optimizer, data, adj_label=None, ir_loss_module=None):


    num_nodes = data.x.size(0)

    if args.data in ["Amazon-ratings", "Roman-empire", "PubMed"]:

        if args.model != "graphmlp" and args.model != "esmlp":
            
            data = get_batch(data, args.batch_size, num_nodes, adj_label)


        else:

            data, adj_label = get_batch(data, args.batch_size, num_nodes, adj_label)

    model.train()
    optimizer.zero_grad()
    labels = data.y
    
    if isinstance(model, GMLP) and adj_label is not None:
        
        loss = train_graph_mlp(args, model, data, labels, adj_label)
        
    elif isinstance(model, ESMLP):
        
        loss = train_es_mlp(epoch, args, model, data, labels, adj_label)

    elif isinstance(model, ESGNN):

        loss = train_esgnn(model, args, data, labels, ir_loss_module)    
    else:
        
        loss = train_basic_model(model, data, labels)

    loss.backward()
    optimizer.step()

    return



def test(model, features, edge_index, num_nodes, labels, val_mask, test_mask, data, args):
    """
    Computes the accuracy on the validation and test mask
    Also needed time for inference is tracked

    In the first block, the whole graph is passed into the model
    In the second block, just the subgraph of the test mask is passed
    """
    model.eval()
    
    """
    First Block 
    """
    start = timeit.default_timer()
    output = model(data)
    end = timeit.default_timer()

    time_whole_graph = end - start
    
    metric = accuracy

    acc_test = metric(output[test_mask], labels[test_mask])
    acc_val = metric(output[val_mask], labels[val_mask])

    acc_test = acc_test.cpu().detach().numpy()
    acc_val = acc_val.cpu().detach().numpy()


    """
    Second Block
    """
    sub_data = data.subgraph(data.test_mask)
    
    start = timeit.default_timer()
    subgraph_output = model(sub_data)
    end = timeit.default_timer()

    time_subgraph = end - start

    subgraph_test_acc = metric(subgraph_output, sub_data.y)
    
    subgraph_test_acc = subgraph_test_acc.cpu().detach().numpy()
    print(acc_test)
    return acc_test, acc_val, time_whole_graph, subgraph_test_acc, time_subgraph



def one_run(args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #data = load_dataset2(args.data, args.split_number)
    #print(is_undirected(data.edge_index))
    data = load_dataset(args.data, seed)

    #data = load_edge_noise_data(args.data, args.distribution, args.number_noise_edges)

    print(args)
    
    if args.model == "mlp":
        model = MLP(
            input_dim=data.x.size(1),
            n_hidden=args.n_hidden,
            n_classes=data.y.max()+1,
            n_layers=args.n_layers,
            dropout=args.dropout,
            activation=F.relu
        )
        
    elif args.model == "gcn":
        model = GCN(
            in_feats=data.x.size(1),
            n_hidden=args.n_hidden,
            n_classes=data.y.max()+1,
            n_layers=args.n_layers,
            dropout=args.dropout,
            activation="relu"
        )
    
    elif args.model == "sage":
        model = GraphSage(
        in_feats=data.x.size(1),
        n_hidden=args.n_hidden,
        n_classes=data.y.max()+1,
        n_layers=args.n_layers,
        dropout=args.dropout,
        activation="relu"
        )
        
    elif args.model == "graphmlp":
        
        model = GMLP(
            nfeat=data.x.size(1),
            nhid=args.n_hidden,
            nclass=data.y.max()+1,
            dropout=args.dropout
        )

    elif args.model == "esmlp":
        
        model = ESMLP(
            nfeat=data.x.size(1),
            nhid=args.n_hidden,
            nclass=data.y.max()+1,
            nlayers=args.n_layers,
            dropout=args.dropout,
            re_eps=args.re_eps,
            ir_eps=args.ir_eps
        )
    
    elif args.model == "esgnn":

        model = ESGNN(
            in_dim=data.x.size(1),
            hidden_dim=args.n_hidden,
            out_dim=data.y.max()+1,
            dropout=args.dropout,
            re_eps=args.re_eps,
            ir_eps=args.ir_eps,
            layer_num=args.n_layers
        )

    elif args.model == "linkx":
        model = LINKX(
            data.x.size(1),
            hidden_channels=args.n_hidden,
            out_channels=data.y.max()+1,
            num_layers=args.n_layers,
            num_nodes=data.x.size(0),  
            init_layers_A=1,
            init_layers_X=1
        )

    
        
    optimizer = optim.Adam(model.parameters(), 
                          lr=args.lr,
                          weight_decay=args.weight_decay)

    print(torch.cuda.is_available())
    if args.cuda and torch.cuda.is_available():
        model.cuda()
        data.cuda()

    """
    Additional object for specific models
    """

    adj_label = None
    ir_loss_module = None
    if args.model == "graphmlp" or args.model == "esmlp":
        #adj_label = getAr(data.edge_index, data.x.size(0), args.order)
        adj_label = get_r_hop_neighborhood(data.edge_index, num_nodes=data.x.size(0), r=args.order)

    elif args.model == "esgnn":
        ir_loss_module = Ir_Consistency_Loss(args.n_hidden // 2, args.dropout)

    model_path = Path(f"model_backup/{args.model}_{args.data}_{args.split_number}_{args.setting_number}_{seed}_backup.pth")
    es = EarlyStopping(patience=20, path=Path(model_path))

    for epoch in tqdm(range(args.epochs)):
        train(
            epoch=epoch,
            args=args,
            model=model,
            data=data,
            optimizer=optimizer,
            adj_label=adj_label,
            ir_loss_module=ir_loss_module
        )



        tmp_test_acc, val_acc, _, _, _ = test(
                            model=model,
                            features=data.x,
                            edge_index=data.edge_index,
                            num_nodes=data.x.size(0),
                            labels=data.y,
                            val_mask=data.val_mask,
                            test_mask=data.test_mask,
                            data=data,
                            args=args
                        )

        if es(val_acc, model):
            model.load_state_dict(torch.load(model_path))
            break



    if args.data=="Cora":
        num_k = [1003, 2006, 4012, 6018, 8024, 10030, 12036, 16048, 20060, 24072, 28036]
    
    else:
        num_k = [20060, 50000, 70000, 100000, 150000, 250000, 350000, 400000]
    test_accuracies = []
    for k in num_k:

        data_copy = data.clone()
        edge_noise_data = heterophile_edge_addition(data_copy.cpu(), k, args.distribution)
        homophily_score = homophily(edge_noise_data.edge_index, edge_noise_data.y)
        edge_noise_data.cuda()


    

    
        #compute final val/test acc
        test_acc, val_acc, time_complete_graph, subgraph_test_acc, time_subgraph = test(
                        model=model,
                        features=data.x,
                        edge_index=edge_noise_data.edge_index,
                        num_nodes=edge_noise_data.x.size(0),
                        labels=edge_noise_data.y,
                        val_mask=edge_noise_data.val_mask,
                        test_mask=edge_noise_data.test_mask,
                        data=edge_noise_data,
                        args=args
                    )
        test_accuracies.append(test_acc)
        #f1_micro, f1_macro = f1_scores(model, data)


    
        results = {
            "test_acc": [test_acc],          # Wrap scalar values in lists if they are scalars
            "K": [k],                        # Same for K
            "Homophily": [homophily_score]   # Same for Homophily
        }


        path = "results/edge_noise/" + args.model + "/" + args.data + "/" + args.distribution + ".csv"
        path = Path(path)
        result_df = pd.DataFrame(results)

        if path.is_file():
            result_df.to_csv(path, mode='a', header=False, index=False)
        else:
            result_df.to_csv(path, mode='a', header=True, index=False)


        
    return results


def load_edge_noise_data(data, distribution, K):
    print(K)
    data = torch.load(f"data/edge_noise_data/{data}/{distribution}/{K}/split_0.pt")
    return data


def run_experiment(args):
    create_setting_folder_edge_noise(args, args.distribution, args.number_noise_edges)
    
    result_list = []
    
    for seed in range(args.n_runs):
        result = one_run(args, seed)

        result_list.append(result)

    
    
    # write_score_edge_noise(result_list, args, args.distribution, args.number_noise_edges)
    

def main():
    run_experiment(args.args)

        

if __name__ == '__main__':
    main()
