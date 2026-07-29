# TODO: XGBoost, LightGBM, Logistic Regression baseline classifiers
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score, classification_report

def train_xgboost(X_train, y_train, config):
    params = config["baseline"]["xgboost"]
    clf = xgb.XGBClassifier(
        n_estimators = params["n_estimators"],
        max_depth = params["max_depth"],
        learning_rate = params["learning_rate"],
        scale_pos_weight = params["scale_pos_weight"],
        random_state = config["training"]["random_seed"],
        eval_metric = "logloss",
        use_label_encoder = False
    )
    clf.fit(X_train,y_train)
    return clf

def train_lightgbm(X_train, y_train, config):
    params = config["baseline"]["lightgbm"]
    clf = lgb.LGBMClassifier(
        n_estimators = params["n_estimators"], 
        num_leaves = params["num_leaves"],
        learning_rate = params["learning_rate"],
        class_weight = params["class_weight"],
        random_state = config["training"]["random_seed"]
    )
    clf.fit(X_train,y_train)
    return clf

def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    y_proba = y_prob[:,1]
    return {
        "f1": f1_score(y_test,y_pred, average = "macro"),
        "auc_roc": roc_auc_score(y_test,y_proba),
        "full report" : classification_report(y_test,y_pred)
    }
    
if __name__ == "__main__":
    from src.data.ingest import load_config, load_raw, build_graph_df
    from src.data.preprocess import get_feature_cols, train_val_test_split, split_known_unknown
    config = load_config("configs/config.yaml")
    classes, edges, features = load_raw(config)
    df = build_graph_df(features, classes, config)
    feature_cols = get_feature_cols(df)
    known_df, unknown_df = split_known_unknown(df)
    X_train,X_val,X_test,y_train,y_val,y_test = train_val_test_split(known_df, feature_cols, config)
    xgb_clf = train_xgboost(X_train, y_train, config)
    results  = evaluate(xgb_clf, X_test, y_test)
    
    """print(f"F1:      {results['f1']:.4f}")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print(results['full report'])"""
    lgb_clf = train_lightgbm(X_train, y_train, config)
    results  = evaluate(lgb_clf, X_test, y_test)
    """print(f"LightGBM F1: {results['f1']:.4f}")
    print(f"LightGBM AUC: {results['auc_roc']:.4f}")"""
    