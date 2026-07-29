# Feature engineering, label splitting, train/val/test split
import re
from sklearn.model_selection import train_test_split

def get_feature_cols(df):
    feature_cols = []
    for i in df.columns:
        if(re.match(r"feat_.+", i)):
            feature_cols.append(i)
    return feature_cols
        
def split_known_unknown(df):
    known_df = df[df["label"] != -1].copy()
    unknown_df = df[df["label"] == -1].copy()
    return known_df, unknown_df


def train_val_test_split(known_df, feature_cols, config):
    X = known_df[feature_cols].values
    y = known_df["label"].values
    seed = config["training"]["random_seed"]
    test_size = config["training"]["test_size"]
    val_size = config["training"]["val_size"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size, random_state = seed, stratify = y)
    val_ratio = val_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_ratio, stratify = y_train, random_state = seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    # Imports here — not at module level — so that importing preprocess.py
    # from another file does NOT trigger CSV loading as a side effect.
    from src.data.ingest import load_config, load_raw, build_graph_df

    config = load_config("configs/config.yaml")
    txs_classes, txs_edges, txs_features = load_raw(config)
    df = build_graph_df(txs_features, txs_classes, config)
    feature_cols = get_feature_cols(df)
    known_df, unknown_df = split_known_unknown(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(known_df, feature_cols, config)
    print(len(df))
    print(len(known_df))
    print(len(unknown_df))
    print(len(feature_cols))
    print(f"{X_train.shape} , {X_val.shape}, {X_test.shape}")
    
if __name__ == "__main__":
    main()