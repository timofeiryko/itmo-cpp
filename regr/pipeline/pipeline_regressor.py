import pandas as pd
from pandarallel import pandarallel
import numpy as np
import re
import joblib
from pathlib import Path
from functools import partial

from rdkit import Chem
from regr.pipeline.regressor_utils import PeptideToSmilesConverter 

from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import Descriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.base import BaseEstimator, TransformerMixin


def compute_protein_analysis_descriptors(seq: str) -> pd.Series:
    """Вычисляет дескрипторы ProteinAnalysis для одной последовательности."""
    import pandas as pd
    import numpy as np
    desc_names = ['MW', 'GRAVY', 'pI', 'Charge', 'Charge_Density', 'Aromaticity', 'Flexibility',
                  'Aliphatic_Index', 'Boman_Index', 'Hydrophobic_AA', 'Polar_AA', 'Positive_AA', 'Negative_AA']
    
    if not isinstance(seq, str) or not seq:
        return pd.Series([np.nan] * len(desc_names), index=desc_names)
    try:
        valid_seq = "".join(re.findall("[A-Z]", seq.upper()))
        if not valid_seq:
            return pd.Series([np.nan] * len(desc_names), index=desc_names)
        analyzer = ProteinAnalysis(valid_seq)
        mw = analyzer.molecular_weight()
        gravy = analyzer.gravy()
        pi = analyzer.isoelectric_point()
        charge = analyzer.charge_at_pH(7.4)
        aromaticity = analyzer.aromaticity()
        flexibility = np.mean(analyzer.flexibility()) if analyzer.flexibility() else np.nan
        aliphatic_index = analyzer.aliphatic_index() 
        boman_index = analyzer.boman()
        charge_density = charge / len(valid_seq) if len(valid_seq) > 0 else 0
        hydrophobic_aa = sum(valid_seq.count(aa) for aa in "AVILMFYW")
        polar_aa = sum(valid_seq.count(aa) for aa in "STNQ")
        positive_aa = sum(valid_seq.count(aa) for aa in "KRH")
        negative_aa = sum(valid_seq.count(aa) for aa in "DE")
        values = [mw, gravy, pi, charge, charge_density, aromaticity, flexibility, aliphatic_index, boman_index,
                  hydrophobic_aa, polar_aa, positive_aa, negative_aa]
        return pd.Series(values, index=desc_names)
    except Exception:
        return pd.Series([np.nan] * len(desc_names), index=desc_names)

def calculate_rdkit_for_row_parallel(smiles, calculator, desc_names):
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
    

class PeptidePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, default_cell_line='HeLa cells'):
        self.default_cell_line = default_cell_line
        print("Инициализация PeptidePreprocessor...")
        self.smiles_converter = PeptideToSmilesConverter()
        desc_names = [desc[0] for desc in Descriptors._descList]
        self.rdkit_calculator = MolecularDescriptorCalculator(desc_names)
        self.rdkit_desc_names = desc_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        print("Начало предобработки...")

        if 'cell_line' not in df.columns:
            df['cell_line'] = self.default_cell_line
        else:
            df.loc[df['cell_line'].isna(), 'cell_line'] = self.default_cell_line
        
        print("Генерация SMILES (параллельно)...")
        df['smiles_sequence'] = df['sequence'].parallel_apply(self.smiles_converter.sequence_to_smiles)
        
        print("Вычисление дескрипторов ProteinAnalysis (параллельно)...")
        protein_descs = df['sequence'].parallel_apply(compute_protein_analysis_descriptors)
        
        print("Вычисление дескрипторов RDKit (параллельно)...")
        partial_func = partial(calculate_rdkit_for_row_parallel, 
                               calculator=self.rdkit_calculator, 
                               desc_names=self.rdkit_desc_names)
        rdkit_descs = df['smiles_sequence'].parallel_apply(partial_func)

        final_df = pd.concat([df, protein_descs, rdkit_descs], axis=1)
        print("Предобработка завершена.")
        return final_df
    
def make_prediction(new_data_df: pd.DataFrame, components_file: str) -> np.ndarray:
    """
    Выполняет полный цикл предсказания для новых данных.
    
    Args:
        new_data_df (pd.DataFrame): DataFrame с обязательным столбцом 'sequence'.
        components_file (str): Путь к файлу .joblib с компонентами пайплайна.
        
    Returns:
        np.ndarray: Массив с предсказанными значениями.
    """
    # 1. Загружаем сохраненные компоненты
    print(f"Загрузка компонентов из {components_file}...")
    components = joblib.load(components_file)
    
    # 2. Создаем и применяем пайплайн предобработки
    preprocessor = PeptidePreprocessor()
    processed_df = preprocessor.transform(new_data_df)

    # 3. Применяем трансформации в том же порядке, что и при обучении
    print("Применение сохраненных трансформаций...")
    
    # 3.1 Подготовка данных: отделяем 'cell_line' от остальных признаков
    # Важно: используем `rdkit_columns` из сохраненного файла, чтобы гарантировать
    # наличие и правильный порядок всех нужных столбцов.
    rdkit_columns = components['rdkit_columns']
    # Проверяем, все ли нужные колонки были сгенерированы
    if not all(col in processed_df.columns for col in rdkit_columns):
        missing = [col for col in rdkit_columns if col not in processed_df.columns]
        raise ValueError(f"В данных после предобработки отсутствуют необходимые столбцы: {missing}")

    rdkit_data = processed_df[rdkit_columns]
    cell_line_data = processed_df[['cell_line']]
    
    # 3.2 One-Hot Encoding для 'cell_line'
    cell_line_encoded = components['ohe_cell_line'].transform(cell_line_data)
    cell_line_df = pd.DataFrame(cell_line_encoded, columns=components['ohe_cell_line'].get_feature_names_out(['cell_line']))
    
    # 3.3 Imputation для RDKit дескрипторов (включая ProteinAnalysis)
    rdkit_imputed = components['imputer_rdkit'].transform(rdkit_data)
    rdkit_df = pd.DataFrame(rdkit_imputed, columns=rdkit_columns)
    
    # 3.4 Объединение признаков
    X_combined = pd.concat([rdkit_df.reset_index(drop=True), cell_line_df.reset_index(drop=True)], axis=1)
    X_combined.columns = X_combined.columns.astype(str) # На всякий случай
    
    # 3.5 Variance Threshold (используем только .transform())
    selected_columns_after_variance = components['selected_columns_after_variance']
    X_var_filtered_array = components['var_selector'].transform(X_combined)
    X_var_filtered = pd.DataFrame(X_var_filtered_array, columns=selected_columns_after_variance)
    
    # 3.6 Scaling
    X_scaled_array = components['scaler'].transform(X_var_filtered)
    
    # 3.7 PCA
    X_pca_array = components['pca'].transform(X_scaled_array)
    
    # 4. Предсказание
    print("Получение предсказаний модели...")
    prediction_log = components['model'].predict(X_pca_array)
    
    # 5. Обратное преобразование (модель обучалась на log1p)
    final_prediction = np.expm1(prediction_log)
    print("Предсказание завершено.")
    
    return final_prediction, processed_df

if __name__ == "__main__":
    pandarallel.initialize(progress_bar=True)
    # Создадим тестовый DataFrame, как будто он пришел "извне"
    # В нем только столбец 'sequence'
    test_data = pd.read_csv('regr/datasets/sequences.csv')

    # Добавим во вторую строку 'cell_line', чтобы проверить оба случая

    print("Исходные данные для предсказания:")
    print(test_data)
    print("-" * 30)

    # Определяем путь к файлу относительно текущего скрипта
    try:
        # Этот блок работает, когда вы запускаете скрипт как файл
        SCRIPT_DIR = Path(__file__).resolve().parent
    except NameError:
        # Этот блок работает, когда вы запускаете код в интерактивной среде (Jupyter)
        # В этом случае ищем файл в текущей рабочей директории
        SCRIPT_DIR = Path.cwd()

    joblib_file = SCRIPT_DIR / "peptide_regressor_artifacts.joblib"

    # Проверка, что файл существует
    if not joblib_file.is_file():
        raise FileNotFoundError(f"Файл артефактов не найден по пути: {joblib_file}")

    # Запускаем полный пайплайн предсказания
    try:
        predictions, result_df = make_prediction(test_data, joblib_file)

        print("-" * 30)
        print("Результаты предсказания (raw_efficiency):")
        
        # Выведем результат в удобном виде
        result_df['predicted_efficiency'] = predictions
        print(result_df[['sequence', 'cell_line', 'predicted_efficiency']])

    except FileNotFoundError:
        print(f"Ошибка: Файл '{joblib_file}' не найден. Убедитесь, что он находится в той же директории или укажите правильный путь.")
    except Exception as e:
        print(f"Произошла ошибка во время выполнения: {e}")