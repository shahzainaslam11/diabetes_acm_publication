from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

def get_model(name, params):
    if name == "rf":
        return RandomForestClassifier(**params)
    elif name == "xgb":
        return xgb.XGBClassifier(**params)
    elif name == "lgbm":
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {name}")
