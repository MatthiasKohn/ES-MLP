import argparse

class args: 
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="esmlp",
                        help='The current setting number per dataset')
    
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    
    parser.add_argument('--setting_number', type=int, default=100000,
                    help='The current setting number per dataset')
    
    parser.add_argument('--split_number', type=int, default=0,
                        help='The current setting number per dataset')
    
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    
    parser.add_argument('--n_runs', type=int, default=10,
                        help='Number of runs per experiment.')
    
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate.')
    
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    
    parser.add_argument('--n_hidden', type=int, default=64,
                        help='Number of hidden units.')
    
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    
    parser.add_argument('--data', type=str, default='Cora',
                        help='dataset to be used')
    
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='To control the ratio of Ncontrast loss')
    
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='batch size')
    
    parser.add_argument('--order', type=int, default=2,
                        help='to compute order-th power of adj')
    
    parser.add_argument('--tau', type=float, default=1.0,
                        help='temperature for Ncontrast loss')
    
    parser.add_argument('--re_eps', type=float, default=0.7,
                        help='Scaling parameter for relevant channel')
    
    parser.add_argument('--ir_eps', type=float, default=0.3,
                        help='Scaling parameter for irrelevant channel')

    parser.add_argument('--beta', type=float, default=0.00001,
                    help='To control the ratio of ICR loss')

    parser.add_argument('--n_layers', type=int, default=2,
                        help='The current setting number per dataset')
    
    parser.add_argument('--model_path', type=str, default="model_backup/",
                    help='Backup_path for Early Stopping')
    
    parser.add_argument('--distribution', type=str, default="Uniform",
                help='Type of Edge Noise')
    
    parser.add_argument('--number_noise_edges', type=int, default=1003,
            help='Type of Edge Noise')
    

    

    args = parser.parse_args()
    
"""
model = "mlp"

data = "Cora"

n_hidden = 64
n_layers = 2
dropout = 0.1

weight_decay=5e-4

reg = 6e-6
lr = 0.06

epochs = 100

n_runs = 10

early = 100

split_number = 0

setting_number=10001

re_eps = 0.1
ir_eps = 0.3
ir_con_lambda = 3e-6



cuda = True
""" 
