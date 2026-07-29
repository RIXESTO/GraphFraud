import pandas as pd
import yaml

def load_config(config_path):
    return yaml.safe_load(open(config_path,"r"))

def load_raw(config):
    classes_path = config["paths"]["raw_classes"]
    edges_path = config["paths"]["raw_edgelist"]
    features_path = config["paths"]["raw_features"]
    
    txs_classes = pd.read_csv(classes_path)
    txs_edges = pd.read_csv(edges_path)
    txs_features = pd.read_csv(features_path, header = None)
    
    txs_features.columns = ["txId","timestep"] + [f"feat_{i}" for i in range(1, txs_features.shape[1]-1)]
    return txs_classes, txs_edges, txs_features

def build_graph_df(features, classes, config): 
    df = features.merge(classes,on= "txId", how= "left")
    lm = config["labels"]
    label_map = {
        "1": lm["illicit"],
        "2": lm["licit"],
        "unknown": lm["unknown"]
    }
    df["label"] = df["class"].map(label_map)
    return df

def main(): 
    config = load_config("configs/config.yaml")
    txs_classes, txs_edges, txs_features = load_raw(config) 
    df = build_graph_df(txs_features, txs_classes, config) 
    for i in df:
        print(i)

if __name__ == "__main__":
    main()

