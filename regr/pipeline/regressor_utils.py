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

import joblib
from functools import partial

# Импорты для RDKit и BioPython
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import Descriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# Импорт для пайплайна sklearn
from sklearn.base import BaseEstimator, TransformerMixin

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
            # Removed invalid 'b' entry to avoid conflicts
        }
        
    @staticmethod
    def expand_repeats(sequence: str) -> str:
        """Expand notation like R8 to RRRRRRRR."""
        # Use a substitution function to replace each occurrence of <letter(s)><digit>
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
            final_smiles = Chem.CanonSmiles(final_smiles, useChiral=True)
            return final_smiles
        except Exception:
            return None
    
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        sequence_col: str = 'sequence',
        output_col: str = 'smiles_sequence'
    ) -> pd.DataFrame:
        """Process a dataframe with peptide sequences."""
        df[output_col] = df[sequence_col].apply(self.sequence_to_smiles)
        return df
    
    def get_bad_sequences_df(self) -> pd.DataFrame:
        """Get a dataframe of problematic sequences."""
        return pd.DataFrame({
            'sequence': list(self.bad_sequences),
            'reason': 'Contains unrecognized components'
        })
    
    def report_validity(self, df: pd.DataFrame, smiles_col: str = 'smiles_sequence') -> Dict:
        """Generate a validity report."""
        valid = df[smiles_col].notna()
        return {
            'valid_percentage': valid.mean() * 100,
            'invalid_count': len(df) - valid.sum(),
            'unrecognized_aa': dict(self.unrecognized_aa_counter)
        }
    
    # ---  ---



# --- Вспомогательные функции, вынесенные из класса ---

def compute_protein_analysis_descriptors(seq: str) -> pd.Series:
    """Вычисляет дескрипторы ProteinAnalysis для одной последовательности."""
    import pandas as pd
    import numpy as np
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    
    desc_names = ['MW', 'GRAVY', 'pI', 'Charge', 'Charge_Density', 'Aromaticity', 'Flexibility',
                  'Aliphatic_Index', 'Boman_Index', 'Hydrophobic_AA', 'Polar_AA', 'Positive_AA', 'Negative_AA']
    
    if not isinstance(seq, str) or not seq:
        return pd.Series([np.nan] * len(desc_names), index=desc_names)
    try:
        valid_seq = "".join(re.findall("[A-Z]", seq.upper()))
        if not valid_seq:
            return pd.Series([np.nan] * len(desc_names), index=desc_names)
        analyzer = ProteinAnalysis(valid_seq)
        values = [
            analyzer.molecular_weight(), analyzer.gravy(), analyzer.isoelectric_point(),
            analyzer.charge_at_pH(7.4), analyzer.charge_at_pH(7.4) / len(valid_seq) if len(valid_seq) > 0 else 0,
            analyzer.aromaticity(), np.mean(analyzer.flexibility()) if analyzer.flexibility() else np.nan,
            analyzer.aliphatic_index(), analyzer.boman(),
            sum(valid_seq.count(aa) for aa in "AVILMFYW"), sum(valid_seq.count(aa) for aa in "STNQ"),
            sum(valid_seq.count(aa) for aa in "KRH"), sum(valid_seq.count(aa) for aa in "DE")
        ]
        return pd.Series(values, index=desc_names)
    except Exception:
        return pd.Series([np.nan] * len(desc_names), index=desc_names)

def calculate_rdkit_for_row_parallel(smiles, calculator, desc_names):
    """Вспомогательная функция для параллельного расчета RDKit дескрипторов."""
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    
    if not isinstance(smiles, str) or not smiles:
        return pd.Series([np.nan] * len(desc_names), index=desc_names)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return pd.Series([np.nan] * len(desc_names), index=desc_names)
        values = calculator.CalcDescriptors(mol)
        return pd.Series(values, index=desc_names)
    except Exception:
        return pd.Series([np.nan] * len(desc_names), index=desc_names)

# --- Класс препроцессора для регрессора ---

class PeptideRegressorPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, default_cell_line='HeLa cells'):
        self.default_cell_line = default_cell_line
        self.smiles_converter = PeptideToSmilesConverter()
        desc_names = [desc[0] for desc in Descriptors._descList]
        self.rdkit_calculator = MolecularDescriptorCalculator(desc_names)
        self.rdkit_desc_names = desc_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        if 'cell_line' not in df.columns:
            df['cell_line'] = self.default_cell_line
        else:
            df.loc[df['cell_line'].isna(), 'cell_line'] = self.default_cell_line
        
        print("Генерация SMILES...")
        df['smiles_sequence'] = df['sequence'].apply(self.smiles_converter.sequence_to_smiles)

        print("Вычисление дескрипторов ProteinAnalysis...")
        protein_descs = df['sequence'].apply(compute_protein_analysis_descriptors)

        print("Вычисление дескрипторов RDKit...")
        parallel_threshold = 100 
        use_parallel = len(df) > parallel_threshold

        if use_parallel:
                print(f"    -> Используется параллельный режим (строк: {len(df)} > {parallel_threshold})")
                apply_method = df['smiles_sequence'].parallel_apply
        else:
                print(f"    -> Используется последовательный режим (строк: {len(df)} <= {parallel_threshold})")
                apply_method = df['smiles_sequence'].apply
        # -------------------------

        partial_func = partial(calculate_rdkit_for_row_parallel, 
                                calculator=self.rdkit_calculator, 
                                desc_names=self.rdkit_desc_names)
            
        # Применяем выбранный метод
        rdkit_descs = apply_method(partial_func)

        final_df = pd.concat([df, protein_descs, rdkit_descs], axis=1)
        return final_df

# --- Основная функция предсказания для регрессора ---

def predict_raw_efficiency(df_to_predict: pd.DataFrame, regressor_artifacts: dict) -> np.ndarray:
    """
    Применяет все шаги трансформации и предсказывает raw_efficiency.
    """
    # 1. Подготовка данных
    rdkit_columns = regressor_artifacts['rdkit_columns']
    if not all(col in df_to_predict.columns for col in rdkit_columns):
        missing = [col for col in rdkit_columns if col not in df_to_predict.columns]
        raise ValueError(f"В данных для регрессора отсутствуют необходимые столбцы: {missing}")

    rdkit_data = df_to_predict[rdkit_columns]
    cell_line_data = df_to_predict[['cell_line']]

    # 2. Трансформации
    cell_line_encoded = regressor_artifacts['ohe_cell_line'].transform(cell_line_data)
    cell_line_df = pd.DataFrame(cell_line_encoded, columns=regressor_artifacts['ohe_cell_line'].get_feature_names_out(['cell_line']))
    
    rdkit_imputed = regressor_artifacts['imputer_rdkit'].transform(rdkit_data)
    rdkit_df = pd.DataFrame(rdkit_imputed, columns=rdkit_columns, index=df_to_predict.index)
    
    X_combined = pd.concat([rdkit_df, cell_line_df.set_index(rdkit_df.index)], axis=1)
    X_combined.columns = X_combined.columns.astype(str)
    
    selected_columns = regressor_artifacts['selected_columns_after_variance']
    X_var_filtered_array = regressor_artifacts['var_selector'].transform(X_combined)
    X_var_filtered = pd.DataFrame(X_var_filtered_array, columns=selected_columns, index=X_combined.index)
    
    X_scaled_array = regressor_artifacts['scaler'].transform(X_var_filtered)
    X_pca_array = regressor_artifacts['pca'].transform(X_scaled_array)
    
    # 3. Предсказание
    prediction_log = regressor_artifacts['model'].predict(X_pca_array)
    final_prediction = np.expm1(prediction_log)
    
    return final_prediction