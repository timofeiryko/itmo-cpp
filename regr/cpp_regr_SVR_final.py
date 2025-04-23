import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import optuna

def convert_to_number(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace(' ', '') 

    if val.startswith('<'):
        num = re.findall(r'<(\d+\.?\d*)', val)
        return float(num[0]) if num else np.nan
    # обработка значений с ±
    elif '±' in val:
        nums = re.findall(r'([\d\.]+)±([\d\.]+)', val)
        if nums:
            main, uncertainty = nums[0]
            return float(main)  
        else:
            return np.nan

    elif '/' in val:
        nums = re.findall(r'([\d\.]+)/([\d\.]+)', val)
        if nums:
            num1, num2 = nums[0]
            return (float(num1) + float(num2)) / 2
        else:
            return np.nan

    else:
        try:
            return float(val)
        except:
            return np.nan


def remove_outliers(df, target_column):
    df_clean = df.copy()

    Q1 = df_clean['raw_efficiency'].quantile(0.25)
    Q3 = df_clean['raw_efficiency'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df_clean[(df_clean['raw_efficiency'] >= lower_bound) & (df_clean['raw_efficiency'] <= upper_bound)]

    return df_clean

# Load dataset
df = pd.read_csv("regr/datasets/for_regr_descriptors_full.csv")
df['raw_efficiency'] = df['raw_efficiency'].apply(convert_to_number)

# Load embeddings
blomap_embeddings = np.load("regr/datasets/blomap_regr.npy")
fingerprints_embeddings = np.load("regr/datasets/fingerprints_regr.npy")
protbert_embeddings = np.load("regr/datasets/protbert_regr.npy")

# Select numerical features
fp_path_index = df.columns.get_loc('fp_path')
selected_features = ['raw_efficiency', 'uptake_type'] + list(df.columns[fp_path_index + 1:])

X_numerical = df[selected_features].copy()

# One-hot encoding for cell_line
if "cell_line" in df.columns:
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cell_line_encoded = enc.fit_transform(df[['cell_line']])
    cell_line_feature_names = enc.get_feature_names_out(["cell_line"])
    X_cell_line = pd.DataFrame(cell_line_encoded, columns=cell_line_feature_names)

# Filter dataframe by ‘uptake_type’ column
X_numerical_filtered = X_numerical[X_numerical['uptake_type'].isin(['Mean Fluorescence intensity', 'Fluorescence intensity'])].copy()

X_numerical_filtered_no_outliers = remove_outliers(X_numerical_filtered, 'raw_efficiency')
filtered_indices = X_numerical_filtered_no_outliers.index

# Filter the embedding arrays
blomap_embeddings_filtered = blomap_embeddings[filtered_indices]
fingerprints_embeddings_filtered = fingerprints_embeddings[filtered_indices]
protbert_embeddings_filtered = protbert_embeddings[filtered_indices]
X_cell_line_filtered = X_cell_line.loc[filtered_indices]

target = 'raw_efficiency'
rdkit_descriptors = X_numerical_filtered_no_outliers.drop(columns=['uptake_type', target])

imputer = SimpleImputer(strategy="mean")
rdkit_descriptors = pd.DataFrame(imputer.fit_transform(rdkit_descriptors))

rdkit_descriptors = rdkit_descriptors.reset_index(drop=True)
blomap_embeddings_filtered = pd.DataFrame(blomap_embeddings_filtered).reset_index(drop=True)
fingerprints_embeddings_filtered = pd.DataFrame(fingerprints_embeddings_filtered).reset_index(drop=True)
protbert_embeddings_filtered = pd.DataFrame(protbert_embeddings_filtered).reset_index(drop=True)
X_cell_line_filtered = pd.DataFrame(X_cell_line_filtered).reset_index(drop=True)

list_of_dfs_named = {
    "rdkit": rdkit_descriptors,
    "blomap": blomap_embeddings_filtered,
    "fingerprints": fingerprints_embeddings_filtered,
    "protbert": protbert_embeddings_filtered,
}

def rename_columns_with_suffix(dfs: dict) -> dict:
    renamed = {}
    for name, df in dfs.items():
        df_ = df.copy()
        df_.columns = [f"{col}_{name}" for col in df.columns]
        renamed[name] = df_
    return renamed

renamed_dfs = rename_columns_with_suffix(list_of_dfs_named)
combined_df_concat = pd.concat(renamed_dfs.values(), axis=1)

dfs_dict = {
    'RDKit_Descriptors': rdkit_descriptors,
}

def apply_varThreshold (X, threshold=0): 
    selector = VarianceThreshold(threshold)
    X_transformed_array = selector.fit_transform(X)
    selected_columns_mask = selector.get_support()
    selected_columns_names = X.columns[selected_columns_mask]
    X_filtered_var = pd.DataFrame(X_transformed_array, columns=selected_columns_names)

    return X_filtered_var

def apply_corr(X, threshold = 0.2):
    correlations = X.apply(lambda col: col.corr(y))
    selected_features = correlations[correlations.abs() >= threshold].index
    X_corr = X[selected_features]

    return X_corr

def apply_scaler (train, test):
    train.columns = train.columns.astype(str)
    test.columns = test.columns.astype(str)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = pd.DataFrame(scaler.fit_transform(train))
    test_scaled = pd.DataFrame(scaler.transform(test))

    return train_scaled, test_scaled

def apply_pca (X_train, X_test, threshold=0.95):
    pca = PCA(n_components=threshold, svd_solver='full')
    train_transformed = pd.DataFrame(pca.fit_transform(X_train))
    test_transformed = pd.DataFrame(pca.transform(X_test))

    return train_transformed, test_transformed

# Training

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)

    print(f'Train MSE: {train_mse:.4f}')
    print(f'Train R2: {train_r2:.4f}')
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test R2: {test_r2:.4f}')

def svr_optuna(X_train, y_train, X_test, y_test):
    # Target function for optimisation
    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['poly', 'rbf'])
    
        params = {
            "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "kernel": kernel,
        }

        if kernel in ['rbf', 'poly']:
            params["gamma"] = trial.suggest_categorical("gamma", ['scale', 'auto'])

        if kernel == 'poly':
            params["degree"] = trial.suggest_int("degree", 2, 5)

        if kernel == 'poly':
            params["coef0"] = trial.suggest_float("coef0", -3.0, 3.0)

        model = SVR(**params)
        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        return score.mean()
    
    sampler = optuna.samplers.TPESampler(seed=8)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100)

    print("The best hyperparameters:", study.best_params)
    print("Best mean error (CV):", -study.best_value)

    best_model = SVR(**study.best_params)
    best_model.fit(X_train, y_train)

    evaluate_model(best_model, X_train, y_train, X_test, y_test)

    return best_model

y = np.log1p(X_numerical_filtered_no_outliers[target]).reset_index(drop=True)

for name, df in dfs_dict.items():
    print(f"\n=== {name} ===")

    X = pd.concat([df, X_cell_line_filtered], axis=1)

    X.columns = pd.Index(X.columns).map(str)
    
    X_filtered_var = apply_varThreshold(X, 0)
    X_train, X_test, y_train, y_test = train_test_split(X_filtered_var, y, test_size=0.2, random_state=42)

    X_train_scaled, X_test_scaled = apply_scaler (X_train, X_test)
    X_train_transformed, X_test_transformed = apply_pca (X_train_scaled, X_test_scaled, 0.95)

    model = svr_optuna(X_train_transformed, y_train,X_test_transformed, y_test)


