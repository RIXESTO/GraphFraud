# TODO: build NetworkX graph and PyTorch Geometric Data object from edgelist
import networkx as nx
import pandas as pd
import torch
import numpy as np
from src.data.ingest import load_config, load_raw, build_graph_df
from torch_geometric.data import Data
def build_networkx_graph(edgelist,df):
    G = nx.from_pandas_edgelist(edgelist,
                                source = "txId1",target = "txId2", 
                                create_using = nx.DiGraph())
    attrs = df.set_index("txId")[["timestep","label"]].to_dict("index")
    nx.set_node_attributes(G,attrs)
    return G
    
def build_pyg_data(df, edgelist, feature_cols):
    node_map = {txId : i for i, txId in enumerate(df["txId"].values)}
    x = torch.tensor(df[feature_cols].values,
                     dtype = torch.float)
    y = torch.tensor(df["label"].values, dtype = torch.long)
    src = edgelist["txId1"].map(node_map)
    dst = edgelist["txId2"].map(node_map)
    valid_mask = src.notna() & dst.notna()
    src = src[valid_mask]
    dst = dst[valid_mask]
    edge_index = torch.tensor(np.stack([src.values, dst.values], axis = 0),
                              dtype = torch.long)
    return Data(x=x, edge_index = edge_index, y=y)


if __name__ == "__main__":
    from src.data.ingest import load_config, load_raw, build_graph_df
    from src.data.preprocess import get_feature_cols
    config = load_config("configs/config.yaml")
    classes, edges, features = load_raw(config)
    df = build_graph_df(features, classes, config)
    G    = build_networkx_graph(edges, df)
    cols = get_feature_cols(df)
    data = build_pyg_data(df, edges, cols)
    
    print(G.number_of_nodes(), G.number_of_edges())
    print(data)