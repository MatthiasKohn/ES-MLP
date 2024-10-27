## The code to the paper Edge-Splitting MLP: Node Classification on Homophilic and Heterophilic Graphs without Message Passing

## Requirements 
Install required packages with:
pip install -r requirements.txt

## Structure
The start point of the code is main.py. The dataset is constructed in load_dataset.py. Models can be found in models folder.

## Running the Code 
To run the code there are several paramters which can be given along.
Available models are: esmlp, gcn, mlp, gcn, sage, graphmlp, esgnn, linkx
Possible paramters for different settings are: --data,--hidden, --epochs, --alpha, --beta, --order, --re_eps, --ir_eps.

To get the results of ES-MLP run for example:
Cora: python main.py --model="esmlp" --data="cora" --epochs=300 alpha=1 --beta=0.001 --order=2 --setting_number=10000
Roman-empire: python main.py --model="esmlp" --data"Roman-empire" alpha=1 --beta=0.00001 --order=2 --setting_number=10000

## Results
The single runs are stored in Results folder

## Other experimemnts 
Robustness Analysis Experiment can be run with:
python experiment_edge_noise.py --model="esmlp" --data="Cora" --distribution="Uniform"

