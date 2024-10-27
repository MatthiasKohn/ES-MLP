import pandas as pd
from pathlib import Path

def write_score(result_list, args, split_number, setting_number, seed):
    args_dict = {key: value for key, value in vars(args).items() if value is not None}
    
    for i, result in enumerate(result_list):
        combined = args_dict | result
        df = pd.DataFrame(combined, index=[i])
        
        path ="results/" + args.model + "/" + args.data + "/split_" + str(split_number) + "/setting_" + str(setting_number) + ".csv"
        path = Path(path)
        
        if path.is_file():
            df.to_csv(path, mode='a', header=False)
        else:
            df.to_csv(path, mode='a', header=True)

        

def create_setting_folder(args, split_number):
    path ="results/" + args.model + "/" + args.data + "/split_" + str(split_number)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    param_path = path / "args"

def create_setting_folder_edge_noise(args, distribution, k):
    path ="results/edge_noise/" + args.model + "/" + args.data + "/" +  distribution
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    param_path = path / "args"


def write_score_edge_noise(result_list, args, distribution, k):
    args_dict = {key: value for key, value in vars(args).items() if value is not None}

    for i, result in enumerate(result_list):
        combined = args_dict | result
        df = pd.DataFrame(combined, index=[i])
        
        path ="results/edge_noise/" + args.model + "/" + args.data + "/" + distribution + "/" + str(k) + ".csv"
        path = Path(path)
        
        if path.is_file():
            df.to_csv(path, mode='a', header=False)
        else:
            df.to_csv(path, mode='a', header=True)



def write_score_es(results, args, number, seed):
    args_dict = {key: value for key, value in vars(args).items() if value is not None}
    combined = args_dict | results
    df = pd.DataFrame(combined, index=[seed])
    
    path ="results/es_mlp/"+ args.data +"/"
    path = Path(path+"setting_"+str(number)+"/"+"setting_"+str(number)+".csv")
    
    if path.is_file():
        df.to_csv(path, mode='a', header=False)
    else:
        df.to_csv(path, mode='a', header=True)

        
def create_setting_folder_es(args, number):
    path ="results/es_mlp/"+ args.data +"/"
    path = Path(path+"setting_"+str(number))
    path.mkdir(parents=True, exist_ok=True)
    param_path = path / "args"
    
    

def create_csbm_folder(p, q, number):
    path ="results/es_mlp/csbm2/"
    path = Path(path+"p_"+ str(p) + "_q_" + str(q)  + "/setting_" + str(number))
    path.mkdir(parents=True, exist_ok=True)
    param_path = path / "args"
    
def write_csbm_score(results, args, number, seed, p, q):
    args_dict = {key: value for key, value in vars(args).items() if value is not None}
    combined = args_dict | results
    df = pd.DataFrame(combined, index=[seed])
    
    path ="results/es_mlp/csbm2/"
    path = Path(path+ "p_"+ str(p) + "_q_" + str(q) +"/setting_" + str(number) +  "/setting_"+ str(number)+".csv")
    
    if path.is_file():
        df.to_csv(path, mode='a', header=False)
    else:
        df.to_csv(path, mode='a', header=True)