import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load dataset
df = pd.read_csv("for_regr_with_descrip.csv")

# Load embeddings
blomap_embeddings = np.load("blomap_regr.npy")
fingerprints_embeddings = np.load("fingerprints_regr.npy")
protbert_embeddings = np.load("protbert_regr.npy")

# Apply PCA to Blomap for XGBoost
pca_blomap = PCA(n_components=10, random_state=42)
blomap_pca = pca_blomap.fit_transform(blomap_embeddings)

# Select numerical features
selected_features = [
    "MW", "GRAVY", "pI", "Charge", "Charge_Density", "Aromaticity",
    "Flexibility", "Aliphatic_Index", "Boman_Index", "Hydrophobic_AA",
    "Polar_AA", "Positive_AA", "Negative_AA", "MolWt", "LogP",
    "TPSA", "HBD", "HBA", "RotBonds", "Rings", "Fsp3"
]
X_numerical = df[selected_features].copy()

# One-hot encoding for cell_line
if "cell_line" in df.columns:
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cell_line_encoded = enc.fit_transform(df[['cell_line']])
    cell_line_feature_names = enc.get_feature_names_out(["cell_line"])
    X_cell_line = pd.DataFrame(cell_line_encoded, columns=cell_line_feature_names)
else:
    X_cell_line = pd.DataFrame()

# Prepare feature matrices
X_xgb = np.hstack([X_numerical, blomap_pca, fingerprints_embeddings, protbert_embeddings])
if not X_cell_line.empty:
    X_xgb = np.hstack([X_xgb, X_cell_line])

X_lgbm = np.hstack([X_numerical, blomap_embeddings, fingerprints_embeddings, protbert_embeddings])
if not X_cell_line.empty:
    X_lgbm = np.hstack([X_lgbm, X_cell_line])

y = df["id_uptake"].values
valid_idx = ~np.isnan(y)
X_xgb, X_lgbm, y = X_xgb[valid_idx], X_lgbm[valid_idx], y[valid_idx]

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_xgb, X_lgbm = imputer.fit_transform(X_xgb), imputer.fit_transform(X_lgbm)

# Log-transform target variable
y = np.log1p(y)

# Train-test split
X_train_xgb, X_test_xgb, y_train, y_test = train_test_split(X_xgb, y, test_size=0.2, random_state=42)
X_train_lgbm, X_test_lgbm, _, _ = train_test_split(X_lgbm, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=754, max_depth=6, learning_rate=0.054886325307314195,
                         subsample=0.9967873263465272, colsample_bytree=0.8645926672674225,
                         random_state=42)
xgb_model.fit(X_train_xgb, y_train)
xgb_pred = np.expm1(xgb_model.predict(X_test_xgb))

# Train LightGBM
lgbm_model = LGBMRegressor(n_estimators=629, learning_rate=0.0114315426267485, num_leaves=77, 
                            min_data_in_leaf=9, max_depth=7, colsample_bytree=0.7, random_state=42)
lgbm_model.fit(X_train_lgbm, y_train)
lgbm_pred = np.expm1(lgbm_model.predict(X_test_lgbm))

# Ensemble predictions (90% XGBoost, 10% LightGBM)
ensemble_pred = (0.9 * xgb_pred + 0.1 * lgbm_pred)

# Evaluate model
print(f"MAE XGBoost: {mean_absolute_error(np.expm1(y_test), xgb_pred):.4f}")
print(f"MAE LightGBM: {mean_absolute_error(np.expm1(y_test), lgbm_pred):.4f}")
print(f"MAE Ensemble (90% XGBoost, 10% LightGBM): {mean_absolute_error(np.expm1(y_test), ensemble_pred):.4f}")
