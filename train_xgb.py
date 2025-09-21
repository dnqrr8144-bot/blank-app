import pandas as pd, numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib

def train_xgb(df: pd.DataFrame, horizon: int, kind="cls"):
    y_col = "y_cls" if kind=="cls" else "y_reg"
    feats = [c for c in df.columns if c.startswith("nv_") or c.startswith("corr_") or c in ["rsi_14","macd_hist","excess_ret_20"]]
    X = df[feats].values
    y = df[y_col].values
    tscv = TimeSeriesSplit(n_splits=5)
    if kind=="cls":
        model = XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
        )
    else:
        model = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8)
    model.fit(X, y)
    joblib.dump((model, feats), f"xgb_{kind}_{horizon}.pkl")
    return model

if __name__ == "__main__":
    base = pd.read_csv("features_base.csv", index_col=0, parse_dates=True)
    for h in [30,60,90,120,180]:
        df = base.copy()
        train_xgb(df, h, "cls")
        train_xgb(df, h, "reg")