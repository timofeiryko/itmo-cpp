import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re
from collections import Counter
from typing import Set, Dict, List
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import optuna

# ==============================================================================
# 1. Custom Transformer: Amino Acid Sequence to SMILES
# ==============================================================================

class PeptideToSmilesConverter:
    """Converts peptide sequences to SMILES with modification handling."""
    
    def __init__(self):
        self._init_maps()
        self.unrecognized_aa_counter = Counter()
        self.bad_sequences: Set[str] = set()
        
    def _init_maps(self):
        """Initialize SMILES mappings for modifications and amino acids."""
        self.modification_map = {
            "Ac-": "CC(=O)",
            "(Acp)": "CC(=O)",  
            "(biotin)": "C1[C@H]2SC(=S)N[C@H]1CCCCC(=O)",  # Correct biotin structure
            "-NH2": "N",
            "Stearyl": "CCCCCCCCCCCCCCCCCC(=O)",  # 18 carbons (C18)
            "Myristoyl": "CCCCCCCCCCCCCC(=O)",    # Correct (C14)
            "Lauroyl": "CCCCCCCCCCCC(=O)",        # Correct (C12)
            "Nspe": "N[C@H](C(C)O)C(=O)",         # Assume correct stereochemistry
            "Nbtg": "N[C@H](C(C)(C)C)C(=O)",      # N-tert-butylglycine
            "Ac": "CC(=O)",
            "Et": "OCC",                          # Ethyl ester (O-linked)
            "Npm": "N1[C@H](C(C)C)C(=O)",         # Verify context if needed
            "Nssb": "N1[C@H](C(C)CC)C(=O)",       # Verify context
            "Mpa": "SCCC(=O)",                    # Mercaptopropionyl (HS-CH2CH2CO-)
            "Cou": "C1=CC(=O)OC2=CC=CC=C12",      # Coumarin (corrected)
            "Xr": "N[C@H](C(C)C)C(=O)",           # D-valine (matches 'v' in aa_map)
            "NII": "C(C)C",                       # N-isopropyl (assumed)
            "PIC": "C1=CC=NC(=C1)CO",             # 4-picolyl (example)
            "IC": "NC1=NC(=O)NC=N1"               # Isocytosine
        }
        
        self.aa_map = {
            # Standard L/D amino acids
            'A': 'N[C@@H](C)C(=O)', 'a': 'N[C@H](C)C(=O)',
            'C': 'N[C@@H](CS)C(=O)', 'c': 'N[C@H](CS)C(=O)',
            'D': 'N[C@@H](CC(=O)O)C(=O)', 'd': 'N[C@H](CC(=O)O)C(=O)',
            'E': 'N[C@@H](CCC(=O)O)C(=O)', 'e': 'N[C@H](CCC(=O)O)C(=O)',
            'F': 'N[C@@H](CC1=CC=CC=C1)C(=O)', 'f': 'N[C@H](CC1=CC=CC=C1)C(=O)',
            'G': 'NCC(=O)', 'g': 'NCC(=O)',
            'H': 'N[C@@H](CC1=CNC=N1)C(=O)', 'h': 'N[C@H](CC1=CNC=N1)C(=O)',
            'I': 'N[C@@H](C(C)CC)C(=O)', 'i': 'N[C@H](C(C)CC)C(=O)',
            'K': 'N[C@@H](CCCCN)C(=O)', 'k': 'N[C@H](CCCCN)C(=O)',
            'L': 'N[C@@H](CC(C)C)C(=O)', 'l': 'N[C@H](CC(C)C)C(=O)',
            'M': 'N[C@@H](CCSC)C(=O)', 'm': 'N[C@H](CCSC)C(=O)',
            'N': 'N[C@@H](CC(=O)N)C(=O)', 'n': 'N[C@H](CC(=O)N)C(=O)',
            'P': 'N1[C@@H](CCC1)C(=O)', 'p': 'N1[C@H](CCC1)C(=O)',
            'Q': 'N[C@@H](CCC(=O)N)C(=O)', 'q': 'N[C@H](CCC(=O)N)C(=O)',
            'R': 'N[C@@H](CCCNC(=N)N)C(=O)', 'r': 'N[C@H](CCCNC(=N)N)C(=O)',
            'S': 'N[C@@H](CO)C(=O)', 's': 'N[C@H](CO)C(=O)',
            'T': 'N[C@@H](C(O)C)C(=O)', 't': 'N[C@H](C(O)C)C(=O)',
            'V': 'N[C@@H](C(C)C)C(=O)', 'v': 'N[C@H](C(C)C)C(=O)',
            'W': 'N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)', 'w': 'N[C@H](CC1=CNC2=CC=CC=C12)C(=O)',
            'Y': 'N[C@@H](CC1=CC=C(O)C=C1)C(=O)', 'y': 'N[C@H](CC1=CC=C(O)C=C1)C(=O)',
            # Special cases
            'X': '*', '?': '*',
            'O': 'N[C@@H](CCCCN)C(=O)',  # Ornithine (corrected side chain)
            'Aib': 'NC(C)(C)C(=O)',      # Aib without chiral center
            'B': 'N[C@@H](CC(=O)N)C(=O)',  # Asn (standard 'B' ambiguity resolved to Asn)
        }
        
    @staticmethod
    def expand_repeats(sequence: str) -> str:
        """Expand notation like R8 to RRRRRRRR."""
        def repl(match):
            token = match.group(1)
            count = int(match.group(2))
            return token * count
        return re.sub(r'([A-Za-z]+)(\d+)', repl, sequence)
    
    def sequence_to_smiles(self, sequence: str) -> str:
        """Convert a raw peptide sequence to SMILES."""
        try:
            # First try RDKit's built-in conversion.
            mol = Chem.MolFromSequence(sequence)
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass
        
        # Fallback to the custom conversion.
        return self._custom_sequence_conversion(sequence)
    
    def _custom_sequence_conversion(self, sequence: str) -> str:
        """Handle non-standard sequences with modifications."""
        smiles_parts = []
        seq = self.expand_repeats(sequence)
        
        # Process modifications first.
        for mod, smi in self.modification_map.items():
            if mod in seq:
                seq = seq.replace(mod, "")
                smiles_parts.append(smi)
        
        # Remove any non-letter characters (such as dashes).
        seq = re.sub(r'[^A-Za-z]', '', seq)
        
        # Process amino acids one-by-one.
        for aa in seq:
            if aa in self.aa_map:
                smiles_parts.append(self.aa_map[aa])
            else:
                self.unrecognized_aa_counter[aa] += 1
                self.bad_sequences.add(sequence)
        
        final_smiles = "".join(smiles_parts)
        try:
            mol = Chem.MolFromSmiles(final_smiles)
            final_smiles = Chem.MolToSmiles(mol) if mol else None
            if not final_smiles:
                return None
            # Canonicalize smiles, important for consistency
            final_smiles = Chem.CanonSmiles(final_smiles, useChiral=True)
            return final_smiles
        except Exception:
            return None

class SequenceToSmilesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_col: str = 'standard_sequence', smiles_col: str = 'smiles_sequence'):
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.converter = PeptideToSmilesConverter()

    def fit(self, X, y=None):
        return self # No fitting required for this transformer

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sequence_col not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{self.sequence_col}' column.")
        
        X_copy = X.copy()
        X_copy[self.smiles_col] = X_copy[self.sequence_col].apply(self.converter.sequence_to_smiles)
        return X_copy

# ==============================================================================
# 1.5. Custom Transformer: Cell Line One-Hot Encoding
# ==============================================================================

class CellLineEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cell_line_col: str = 'cell_line', default_value: str = 'HeLa cells'):
        self.cell_line_col = cell_line_col
        self.default_value = default_value
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        
        # 1. Handle missing cell_line column: if not present, create and fill
        if self.cell_line_col not in X_copy.columns:
            X_copy[self.cell_line_col] = self.default_value
        
        # 2. Handle empty/NaN values in cell_line column
        # Fill None, NaN, empty strings with default_value
        X_copy[self.cell_line_col] = X_copy[self.cell_line_col].replace('', np.nan).fillna(self.default_value)
        
        # Fit the OneHotEncoder on the cell_line column
        self.encoder.fit(X_copy[[self.cell_line_col]])
        self.feature_names_out_ = self.encoder.get_feature_names_out([self.cell_line_col]).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder is None:
            raise RuntimeError("fit() must be called before transform().")
            
        X_copy = X.copy()
        
        # 1. Handle missing cell_line column: if not present, create and fill
        if self.cell_line_col not in X_copy.columns:
            X_copy[self.cell_line_col] = self.default_value
        
        # 2. Handle empty/NaN values in cell_line column
        # Fill None, NaN, empty strings with default_value
        X_copy[self.cell_line_col] = X_copy[self.cell_line_col].replace('', np.nan).fillna(self.default_value)
        
        # Transform the cell_line column
        cell_line_encoded = self.encoder.transform(X_copy[[self.cell_line_col]])
        df_cell_line_encoded = pd.DataFrame(cell_line_encoded, 
                                            columns=self.feature_names_out_, 
                                            index=X_copy.index)
        
        # Drop the original cell_line column and concatenate with encoded features
        X_processed = X_copy.drop(columns=[self.cell_line_col], errors='ignore') # 'errors=ignore' handles if it was already dropped
        X_processed = pd.concat([X_processed, df_cell_line_encoded], axis=1)
        
        return X_processed

# ==============================================================================
# 2. Custom Transformer: Descriptor Calculation
# ==============================================================================

all_rdkit_descriptors = Descriptors.descList
# *** ЭТА ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ (список строк) ПИКЛЕВАЕМА ***
rdkit_descriptor_names = [desc[0] for desc in all_rdkit_descriptors] 

def compute_sequence_descriptors(seq):
    """aa sequence descriptors"""
    try:
        seq = seq.upper()
        analyzer = ProteinAnalysis(seq)
        mw = analyzer.molecular_weight()
        gravy = analyzer.gravy()
        pi = analyzer.isoelectric_point()
        charge = analyzer.charge_at_pH(7.4)
        aromaticity = analyzer.aromaticity()
        flexibility = np.mean(analyzer.flexibility()) if analyzer.flexibility() else np.nan
        aliphatic_index = sum(analyzer.count_amino_acids().get(aa, 0) for aa in "AVILMF") / len(seq)
        # Boman_Index is typically sum of secondary_structure_fraction for alpha, beta, turn, not just sum()
        # For simplicity, if not specific use-case is given, recalculate or use specific fractions
        # Based on original code: sum(analyzer.secondary_structure_fraction())
        boman_index = sum(analyzer.secondary_structure_fraction()) # This is unusual, double-check source
        charge_density = charge / len(seq)
        hydrophobic_aa = sum(seq.count(aa) for aa in "AVILMFYW")
        polar_aa = sum(seq.count(aa) for aa in "STNQ")
        positive_aa = sum(seq.count(aa) for aa in "KRH")
        negative_aa = sum(seq.count(aa) for aa in "DE")

        return [mw, gravy, pi, charge, charge_density, aromaticity, flexibility, aliphatic_index, boman_index,
                hydrophobic_aa, polar_aa, positive_aa, negative_aa]
    except Exception as e:
        # print(f"mistake seq {seq}: {e}") # Suppress print during pipeline run
        return [np.nan] * 13

def compute_chemical_descriptors(smiles):
    """chem descriptors SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(rdkit_descriptor_names) # Используем длину списка имен

        values = []
        for name in rdkit_descriptor_names:
            func = getattr(Descriptors, name) # Получаем функцию по ее имени
            try:
                val = func(mol)
            except Exception: # Catch any error during descriptor calculation for a specific molecule
                val = np.nan
            values.append(val)

        return values
    except Exception as e:
        # print(f"mistake SMILES '{smiles}': {e}") # Suppress print during pipeline run
        return [np.nan] * len(rdkit_descriptor_names) # Используем длину списка имен

class DescriptorCalculatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_col: str = 'standard_sequence', smiles_col: str = 'smiles_sequence'):
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col

        self.seq_desc_names = ['MW', 'GRAVY', 'pI', 'Charge', 'Charge_Density', 'Aromaticity', 'Flexibility',
                               'Aliphatic_Index', 'Boman_Index', 'Hydrophobic_AA', 'Polar_AA', 'Positive_AA', 'Negative_AA']
        self.chem_desc_names = rdkit_descriptor_names
        self.feature_names_out_ = None # To store final column names after fit

    def fit(self, X, y=None):
        # We need to simulate the transform to get feature names
        _ = self.transform(X.head(1)) # Use a small subset to avoid heavy computation
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sequence_col not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{self.sequence_col}' column.")
        if self.smiles_col not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{self.smiles_col}' column.")

        # Compute sequence descriptors
        sequence_descriptors = X[self.sequence_col].apply(compute_sequence_descriptors)
        df_seq_desc = pd.DataFrame(sequence_descriptors.tolist(), columns=self.seq_desc_names, index=X.index)

        # Compute chemical descriptors
        chemical_descriptors = X[self.smiles_col].apply(compute_chemical_descriptors)
        df_chem_desc = pd.DataFrame(chemical_descriptors.tolist(), columns=self.chem_desc_names, index=X.index)

        # Concatenate all generated descriptors
                # *** ИЗМЕНЕНИЕ: Теперь мы объединяем новые дескрипторы со ВСЕМИ другими колонками из X,
        # которые не являются sequence_col или smiles_col. Это включает OHE-колонки. ***
        
        # Колонки, которые не должны быть преобразованы этим трансформером, но должны быть переданы дальше.
        # Это включает OHE колонки клеточной линии, а также любые другие числовые признаки, если они есть.
        passthrough_cols = [col for col in X.columns if col not in [self.sequence_col, self.smiles_col]]
        df_passthrough = X[passthrough_cols].copy()

        final_features_df = pd.concat([df_chem_desc.reset_index(drop=True), 
                                       df_seq_desc.reset_index(drop=True),
                                       df_passthrough.reset_index(drop=True)], axis=1)

        final_features_df.columns = final_features_df.columns.astype(str)
        self.feature_names_out_ = final_features_df.columns.tolist() 
        return final_features_df

# ==============================================================================
# 3. Custom Transformer: Variance Threshold (adapted from your apply_varThreshold)
# ==============================================================================

class VarianceThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.selector = None # Will store the fitted VarianceThreshold object
        self.features_to_keep_ = None # For storing the names of features that passed the threshold

    def fit(self, X: pd.DataFrame, y=None):
        # Ensure X is DataFrame for column operations, even if it's numpy array
        X_df = pd.DataFrame(X) 
        self.selector = VarianceThreshold(self.threshold)
        self.selector.fit(X_df) # Fit on the actual data
        selected_columns_mask = self.selector.get_support()
        self.features_to_keep_ = X_df.columns[selected_columns_mask].tolist()
        
        if not self.features_to_keep_ and X_df.shape[1] > 0:
            raise ValueError(f"No features meet the variance threshold ({self.threshold}). "
                             "Adjust threshold or check data. All features might have zero variance.")
        return self

    def transform(self, X):
        if self.selector is None or self.features_to_keep_ is None:
            raise RuntimeError("fit() method must be called before transform().")
        
        # Transform using the fitted selector
        X_transformed_array = self.selector.transform(X)
        # Reconstruct DataFrame with original column names
        return X_transformed_array

# ==============================================================================
# 4. Evaluation Function (kept for consistency, not part of pipeline itself)
# ==============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Ensure X_train/X_test are in the format expected by the model (e.g., DataFrame)
    # The pipeline will handle this, so X_train, X_test here are the *raw* inputs.
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

# ==============================================================================
# 5. Optuna Function for Pipeline Optimization
# ==============================================================================

def svr_optuna_pipeline(X_train_processed, y_train, X_test_processed, y_test):
    """
    Optimizes a full preprocessing and SVR pipeline using Optuna.
    X_train_processed, X_test_processed are the *intermediate processed* DataFrames (numerical features).
    """
    def objective(trial):
        var_threshold = trial.suggest_float("var_threshold__threshold", 0.0, 0.1) 
        svr_kernel = trial.suggest_categorical('svr__kernel', ['poly', 'rbf'])
        svr_params = {
            "svr__C": trial.suggest_float("svr__C", 1e-1, 1e3, log=True),
            "svr__epsilon": trial.suggest_float("svr__epsilon", 1e-3, 1.0, log=True),
            "svr__kernel": svr_kernel,
        }

        if svr_kernel in ['rbf', 'poly']:
            svr_params["svr__gamma"] = trial.suggest_categorical("svr__gamma", ['scale', 'auto'])

        if svr_kernel == 'poly':
            svr_params["svr__degree"] = trial.suggest_int("svr__degree", 2, 5)
            svr_params["svr__coef0"] = trial.suggest_float("svr__coef0", -3.0, 3.0)

        # --- Define the Pipeline for Optuna (starts after feature generation) ---
        pipeline_for_optuna = Pipeline([
            # *** Пайплайн начинается с Imputer, так как признаки уже сгенерированы ***
            ('imputer', SimpleImputer(strategy='mean')), 
            ('var_threshold', VarianceThresholdTransformer(threshold=var_threshold)), 
            ('scaler', MinMaxScaler(feature_range=(0,1))), 
            ('pca', PCA(n_components=0.95)), 
            ('svr', SVR())
        ])

        pipeline_for_optuna.set_params(**svr_params)

        score = cross_val_score(
            pipeline_for_optuna, # Используем этот сокращенный пайплайн
            X_train_processed, # И уже обработанные данные
            y_train,
            cv=3, 
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        return score.mean()
    
    sampler = optuna.samplers.TPESampler(seed=8)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # *** Передаем уже обработанные данные для оптимизации ***
    study.optimize(objective, n_trials=10) 

    print("\nBest hyperparameters:", study.best_params)
    print("Best mean error (CV):", -study.best_value)

    best_params = study.best_params
    
    final_var_threshold = best_params.get("var_threshold__threshold", 0.0)
    final_svr_params = {k.replace('svr__', ''): v for k, v in best_params.items() if k.startswith('svr__')}

    # *** ВОССТАНАВЛИВАЕМ ПОЛНЫЙ ПАЙПЛАЙН ДЛЯ СОХРАНЕНИЯ И БУДУЩИХ ПРЕДСКАЗАНИЙ ***
    # Он должен содержать все шаги, чтобы принимать сырые входные данные
    full_pipeline_for_deployment = Pipeline([
        ('seq_to_smiles', SequenceToSmilesTransformer(sequence_col='standard_sequence', smiles_col='smiles_sequence')),
        ('cell_line_encoder', CellLineEncoderTransformer(cell_line_col='cell_line', default_value='HeLa cells')),
        ('descriptor_calc', DescriptorCalculatorTransformer(
            sequence_col='standard_sequence', smiles_col='smiles_sequence')),
        ('imputer', SimpleImputer(strategy='mean')), # Используем strategy='median'
        ('var_threshold', VarianceThresholdTransformer(threshold=final_var_threshold)),
        ('scaler', MinMaxScaler(feature_range=(0,1))),
        ('pca', PCA(n_components=0.95)),
        ('svr', SVR(**final_svr_params))
    ])

    # Обучаем ПОЛНЫЙ пайплайн на СЫРЫХ ТРЕНИРОВОЧНЫХ данных
    full_pipeline_for_deployment.fit(X_train_raw, y_train)

    # Оцениваем ПОЛНЫЙ пайплайн на СЫРЫХ ТЕСТОВЫХ данных
    print("\n--- Final Pipeline Evaluation ---")
    evaluate_model(full_pipeline_for_deployment, X_train_raw, y_train, X_test_raw, y_test)

    return full_pipeline_for_deployment


# ==============================================================================
# Main Training and Saving Logic
# ==============================================================================

if __name__ == "__main__":
    initial_peptides_df = pd.read_csv('regr/datasets/sequences.csv')
    y = pd.read_csv('regr/datasets/y.csv')
    cell_line_data_df = pd.read_csv('regr/datasets/X_cell_line_filtered.csv')
    
    if 'sequence' in initial_peptides_df.columns and 'standard_sequence' not in initial_peptides_df.columns:
        initial_peptides_df.rename(columns={'sequence': 'standard_sequence'}, inplace=True)
    elif 0 in initial_peptides_df.columns and 'standard_sequence' not in initial_peptides_df.columns:
        initial_peptides_df.rename(columns={0: 'standard_sequence'}, inplace=True)
    elif 'standard_sequence' not in initial_peptides_df.columns:
        raise ValueError("The 'sequences.csv' file does not contain a 'standard_sequence' column "
                         "or a column that can be easily renamed to it. "
                         "Please check your CSV file's column names.")
    
    initial_peptides_df = initial_peptides_df.reset_index(drop=True)
    cell_line_data_df = cell_line_data_df.reset_index(drop=True)

    if len(initial_peptides_df) != len(cell_line_data_df):
        raise ValueError("initial_peptides_df and cell_line_data_df must have the same number of rows to be concatenated by index.")

    # X_raw теперь содержит 'standard_sequence', 'cell_line' (если есть) и другие признаки из X_cell_line_filtered
    X_raw = pd.concat([initial_peptides_df, cell_line_data_df], axis=1)
    
    # Убедимся, что колонка 'cell_line' (если она есть) имеет строковый тип, чтобы OneHotEncoder не выдал ошибку
    if 'cell_line' in X_raw.columns:
        X_raw['cell_line'] = X_raw['cell_line'].astype(str)
        
    # Ensure y matches the length of X_raw
    if len(y) > len(X_raw):
        y = y.iloc[:len(X_raw)]
    elif len(y) < len(X_raw):
        raise ValueError("Target variable y has fewer samples than input X_raw.")

    # Split raw data for training the pipeline
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    print("Starting pipeline training...")
    model_name = 'SVR_Pipeline'

    print("Generating intermediate features...")
    feature_generation_pipeline = Pipeline([
        ('seq_to_smiles', SequenceToSmilesTransformer(sequence_col='standard_sequence', smiles_col='smiles_sequence')),
        ('cell_line_encoder', CellLineEncoderTransformer(cell_line_col='cell_line', default_value='HeLa cells')),
        ('descriptor_calc', DescriptorCalculatorTransformer(
            sequence_col='standard_sequence', smiles_col='smiles_sequence')),
    ])

    # Обучаем и трансформируем на тренировочных данных
    X_train_intermediate_features = feature_generation_pipeline.fit_transform(X_train_raw, y_train)
    # Трансформируем тестовые данные (НЕ fit_transform)
    X_test_intermediate_features = feature_generation_pipeline.transform(X_test_raw)
    
    print("Intermediate features generated. Starting Optuna optimization...")

    # *** Передаем уже обработанные признаки в Optuna ***
    # Обратите внимание, что X_train_raw и X_test_raw (оригинальные) 
    # все еще доступны для финальной подгонки full_pipeline_for_deployment.
    full_pipeline_for_deployment = svr_optuna_pipeline(
        X_train_intermediate_features, y_train, X_test_intermediate_features, y_test
    )

    output_dir = 'trained_pipelines'
    os.makedirs(output_dir, exist_ok=True)
    pipeline_filename = os.path.join(output_dir, f'{model_name}.joblib')
    joblib.dump(full_pipeline_for_deployment, pipeline_filename)
    print(f"\nFull pipeline '{model_name}' successfully saved as '{pipeline_filename}'")