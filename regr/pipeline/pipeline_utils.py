import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re
from collections import Counter
from typing import Set, Dict, List, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error


# ==============================================================================
# Utility Functions (shared)
# ==============================================================================

def convert_to_number(val: Any) -> float:
    """
    Converts various string representations of numbers into a float.
    Handles 'NaN', '<' (less than), '±' (plus/minus), and '/' (range) formats.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).replace(' ', '') 

    if val.startswith('<'):
        num = re.findall(r'<(\d+\.?\d*)', val)
        return float(num[0]) if num else np.nan
    
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

def remove_outliers(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Removes outliers from a specified target column using the Interquartile Range (IQR) method.
    Rows where the target column value falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are removed.
    """
    df_clean = df.copy()

    Q1 = df_clean[target_column].quantile(0.25)
    Q3 = df_clean[target_column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df_clean[(df_clean[target_column] >= lower_bound) & (df_clean[target_column] <= upper_bound)]

    return df_clean

def evaluate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
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
# Custom Transformer: Amino Acid Sequence to SMILES
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
            mol = Chem.MolFromSequence(sequence)
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass
        
        return self._custom_sequence_conversion(sequence)
    
    def _custom_sequence_conversion(self, sequence: str) -> str:
        """Handle non-standard sequences with modifications."""
        smiles_parts = []
        seq = self.expand_repeats(sequence)
        
        for mod, smi in self.modification_map.items():
            if mod in seq:
                seq = seq.replace(mod, "")
                smiles_parts.append(smi)
        
        seq = re.sub(r'[^A-Za-z]', '', seq)
        
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
        return self 

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X_copy = X.copy()

        if self.sequence_col not in X_copy.columns:
            # Если нет, проверяем, есть ли колонка 'sequence'
            if 'sequence' in X_copy.columns:
                # Если 'sequence' есть, переименовываем её в self.sequence_col
                X_copy.rename(columns={'sequence': self.sequence_col}, inplace=True)
                print(f"DEBUG: Renamed 'sequence' column to '{self.sequence_col}' in SequenceToSmilesTransformer.")
            else:
                # Если нет ни self.sequence_col, ни 'sequence', вызываем ошибку
                raise ValueError(f"Input DataFrame must contain a '{self.sequence_col}' column "
                                 f"or a 'sequence' column that can be renamed to it.")
        
        
        X_copy[self.smiles_col] = X_copy[self.sequence_col].apply(self.converter.sequence_to_smiles)
        return X_copy

# ==============================================================================
# Custom Transformer: Cell Line One-Hot Encoding
# ==============================================================================

class CellLineEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cell_line_col: str = 'cell_line', default_value: str = 'HeLa cells'):
        self.cell_line_col = cell_line_col
        self.default_value = default_value
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        
        if self.cell_line_col not in X_copy.columns:
            X_copy[self.cell_line_col] = self.default_value
        
        X_copy[self.cell_line_col] = X_copy[self.cell_line_col].replace('', np.nan).fillna(self.default_value).astype(str)
        
        self.encoder.fit(X_copy[[self.cell_line_col]])
        self.feature_names_out_ = self.encoder.get_feature_names_out([self.cell_line_col]).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder is None:
            raise RuntimeError("fit() must be called before transform().")
            
        X_copy = X.copy()
        
        if self.cell_line_col not in X_copy.columns:
            X_copy[self.cell_line_col] = self.default_value
        
        X_copy[self.cell_line_col] = X_copy[self.cell_line_col].replace('', np.nan).fillna(self.default_value).astype(str)
        
        cell_line_encoded = self.encoder.transform(X_copy[[self.cell_line_col]])
        df_cell_line_encoded = pd.DataFrame(cell_line_encoded, 
                                            columns=self.feature_names_out_, 
                                            index=X_copy.index)
        
        X_processed = X_copy.drop(columns=[self.cell_line_col], errors='ignore') 
        numerical_cols_to_keep = X_processed.select_dtypes(include=np.number).columns.tolist()
        X_processed = pd.concat([X_processed[numerical_cols_to_keep], df_cell_line_encoded], axis=1)
        
        return X_processed

# ==============================================================================
# Custom Transformer: Descriptor Calculation
# ==============================================================================

all_rdkit_descriptors = Descriptors.descList
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
        boman_index = sum(analyzer.secondary_structure_fraction()) 
        charge_density = charge / len(seq)
        hydrophobic_aa = sum(seq.count(aa) for aa in "AVILMFYW")
        polar_aa = sum(seq.count(aa) for aa in "STNQ")
        positive_aa = sum(seq.count(aa) for aa in "KRH")
        negative_aa = sum(seq.count(aa) for aa in "DE")

        return [mw, gravy, pi, charge, charge_density, aromaticity, flexibility, aliphatic_index, boman_index,
                hydrophobic_aa, polar_aa, positive_aa, negative_aa]
    except Exception as e:
        return [np.nan] * 13

def compute_chemical_descriptors(smiles):
    """chem descriptors SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(rdkit_descriptor_names)

        values = []
        for name in rdkit_descriptor_names:
            func = getattr(Descriptors, name)
            try:
                val = func(mol)
            except Exception: 
                val = np.nan
            values.append(val)

        return values
    except Exception as e:
        return [np.nan] * len(rdkit_descriptor_names)

class DescriptorCalculatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_col: str = 'standard_sequence', smiles_col: str = 'smiles_sequence'):
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col

        self.seq_desc_names = ['MW', 'GRAVY', 'pI', 'Charge', 'Charge_Density', 'Aromaticity', 'Flexibility',
                               'Aliphatic_Index', 'Boman_Index', 'Hydrophobic_AA', 'Polar_AA', 'Positive_AA', 'Negative_AA']
        self.chem_desc_names = rdkit_descriptor_names
        self.feature_names_out_ = None 

    def fit(self, X, y=None):
        _ = self.transform(X.head(1)) 
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sequence_col not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{self.sequence_col}' column.")
        if self.smiles_col not in X.columns:
            raise ValueError(f"Input DataFrame must contain a '{self.smiles_col}' column.")

        sequence_descriptors = X[self.sequence_col].apply(compute_sequence_descriptors)
        df_seq_desc = pd.DataFrame(sequence_descriptors.tolist(), columns=self.seq_desc_names, index=X.index)

        chemical_descriptors = X[self.smiles_col].apply(compute_chemical_descriptors)
        df_chem_desc = pd.DataFrame(chemical_descriptors.tolist(), columns=self.chem_desc_names, index=X.index)

        passthrough_cols = [col for col in X.columns if col not in [self.sequence_col, self.smiles_col]]
        df_passthrough = X[passthrough_cols].copy()

        final_features_df = pd.concat([df_chem_desc.reset_index(drop=True), 
                                       df_seq_desc.reset_index(drop=True),
                                       df_passthrough.reset_index(drop=True)], axis=1)

        final_features_df.columns = final_features_df.columns.astype(str)
        self.feature_names_out_ = final_features_df.columns.tolist() 
        return final_features_df

# ==============================================================================
# Custom Transformer: Variance Threshold
# ==============================================================================

class VarianceThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.selector = None 
        self.features_to_keep_ = None

    def fit(self, X, y=None): 
        # Determine column names for feature_names_out_
        # Ensure input_cols is a Pandas Index or NumPy array for boolean indexing
        if isinstance(X, pd.DataFrame):
            input_cols = X.columns
        else: # If X is a numpy array, create generic names as a Pandas Index
            input_cols = pd.Index([f'feature_{i}' for i in range(X.shape[1])])
            
        self.selector = VarianceThreshold(self.threshold)
        self.selector.fit(X) 
        selected_columns_mask = self.selector.get_support()
        
        self.features_to_keep_ = input_cols[selected_columns_mask].tolist()

        if not self.features_to_keep_ and X.shape[1] > 0: 
            raise ValueError(f"No features meet the variance threshold ({self.threshold}). "
                             "Adjust threshold or check data. All features might have zero variance.")
        return self

    def transform(self, X): 
        if self.selector is None or self.features_to_keep_ is None:
            raise RuntimeError("fit() method must be called before transform().")
        
        X_transformed_array = self.selector.transform(X)
        return X_transformed_array 