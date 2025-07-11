import numpy as np
import pandas as pd
import re
import os
import joblib
import optuna

# Импортируем все необходимые классы и функции из нашего нового файла utilities
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score

from pipeline_utils import (
    convert_to_number, remove_outliers, evaluate_model,
    SequenceToSmilesTransformer, CellLineEncoderTransformer, 
    DescriptorCalculatorTransformer, VarianceThresholdTransformer
)

# ==============================================================================
# 1. Загрузка данных и предварительная обработка (из старого кода для обучения трансформаций)
# ==============================================================================

# Загружаем основной датасет, который использовался в старом коде
df_old_data = pd.read_csv("regr/datasets/for_regr_descriptors_full.csv")

# Преобразование raw_efficiency (как в старом коде)
df_old_data['raw_efficiency'] = df_old_data['raw_efficiency'].apply(convert_to_number)

# Отбор по uptake_type (как в старом коде)
df_filtered_uptake = df_old_data[df_old_data['uptake_type'].isin(['Mean Fluorescence intensity', 'Fluorescence intensity'])].copy()

# Удаление выбросов (как в старом коде)
df_filtered_outliers = remove_outliers(df_filtered_uptake, 'raw_efficiency')

# Целевая переменная y (как в старом коде)
y_original_target = np.log1p(df_filtered_outliers['raw_efficiency']).reset_index(drop=True)

# Идентификация колонок RDKit дескрипторов (как в старом коде)
# Они уже предварительно рассчитаны в 'for_regr_descriptors_full.csv'
fp_path_index = df_old_data.columns.get_loc('fp_path')
rdkit_descriptor_cols = df_old_data.columns[fp_path_index + 1:].tolist()

# Извлечение RDKit дескрипторов из уже отфильтрованного DataFrame
# (эти дескрипторы будут использоваться для обучения импутера, селектора, скалера и PCA)
rdkit_descriptors_for_training = df_filtered_outliers[rdkit_descriptor_cols].copy()

# Обработка cell_line для обучения OneHotEncoder
# В старом коде OneHotEncoder обучался на всем df['cell_line'] (до фильтрации по uptake/outliers)
# Затем отфильтровывался по filtered_indices.
# Мы будем обучать CellLineEncoderTransformer на полном наборе,
# а потом применять его к отфильтрованному X.

# Для получения обученного энкодера:
# Создадим временный DataFrame с полной колонкой cell_line из исходных данных (df_old_data)
# Это важно, чтобы энкодер увидел ВСЕ возможные категории cell_line.
full_cell_line_data_for_encoder_fit = df_old_data[['cell_line']].copy()
# Ensure it's string type for the encoder
full_cell_line_data_for_encoder_fit['cell_line'] = full_cell_line_data_for_encoder_fit['cell_line'].replace('', np.nan).fillna('HeLa cells').astype(str)

# Обучаем CellLineEncoderTransformer
fitted_cell_line_encoder = CellLineEncoderTransformer(cell_line_col='cell_line', default_value='HeLa cells')
fitted_cell_line_encoder.fit(full_cell_line_data_for_encoder_fit)

# Применяем обученный энкодер к отфильтрованным данным (по индексам df_filtered_outliers)
# Чтобы получить OHE-признаки клеточной линии, соответствующие отфильтрованным строкам
X_cell_line_filtered_ohe = fitted_cell_line_encoder.transform(df_filtered_outliers[['cell_line']].copy())


# Комбинируем RDKit дескрипторы и One-Hot кодированные признаки клеточной линии
# для обучения последующих трансформаций (Imputer, VarThreshold, Scaler, PCA)
X_combined_features_for_old_pipeline_training = pd.concat([
    rdkit_descriptors_for_training.reset_index(drop=True),
    X_cell_line_filtered_ohe.reset_index(drop=True)
], axis=1)

# Ensure column names are strings for consistency
X_combined_features_for_old_pipeline_training.columns = X_combined_features_for_old_pipeline_training.columns.astype(str)

# Разделение данных для обучения старых трансформаций и SVR
X_train_for_old_transforms, X_test_for_old_transforms, y_train, y_test = train_test_split(
    X_combined_features_for_old_pipeline_training, y_original_target, test_size=0.2, random_state=42
)

print("Данные подготовлены для обучения трансформаций старого кода.")

# ==============================================================================
# 2. Обучение и сохранение всех трансформаций из старого кода
# ==============================================================================

# 1. Imputer (из старого кода: strategy='mean')
fitted_imputer = SimpleImputer(strategy="mean")
X_train_imputed = fitted_imputer.fit_transform(X_train_for_old_transforms)
X_test_imputed = fitted_imputer.transform(X_test_for_old_transforms)

# 2. VarianceThreshold (из старого кода: threshold=0)
# Используем наш кастомный VarianceThresholdTransformer, но обучим его здесь
fitted_var_selector = VarianceThresholdTransformer(threshold=0)
X_train_var_filtered = fitted_var_selector.fit_transform(X_train_imputed)
X_test_var_filtered = fitted_var_selector.transform(X_test_imputed)

# 3. MinMaxScaler (из старого кода: default range)
fitted_scaler = MinMaxScaler(feature_range=(0,1)) # Default is (0,1) anyway
X_train_scaled = fitted_scaler.fit_transform(X_train_var_filtered)
X_test_scaled = fitted_scaler.transform(X_test_var_filtered)

# 4. PCA (из старого кода: n_components=0.95)
fitted_pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca_transformed = fitted_pca.fit_transform(X_train_scaled)
X_test_pca_transformed = fitted_pca.transform(X_test_scaled)

print("Все индивидуальные трансформации из старого кода обучены.")

# ==============================================================================
# 3. Оптимизация SVR (используем функцию из старого кода, адаптированную)
# ==============================================================================

# Функция svr_optuna из старого кода (скопирована сюда, так как она обучает SVR)
def svr_optuna_old_code_style(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series) -> SVR:
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
            params["coef0"] = trial.suggest_float("coef0", -3.0, 3.0)

        model = SVR(**params)
        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5, # В старом коде cv=5
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        return score.mean()
    
    sampler = optuna.samplers.TPESampler(seed=8)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100) # В старом коде n_trials=100

    print("The best hyperparameters for SVR:", study.best_params)
    print("Best mean error (CV):", -study.best_value)

    best_model = SVR(**study.best_params)
    best_model.fit(X_train, y_train)

    evaluate_model(best_model, X_train, y_train, X_test, y_test)

    return best_model

print("\nЗапускаем оптимизацию SVR для получения обученной модели...")
fitted_svr_model = svr_optuna_old_code_style(X_train_pca_transformed, y_train, X_test_pca_transformed, y_test)

print("SVR модель обучена с лучшими параметрами.")

# ==============================================================================
# 4. Сборка и сохранение финального деплоябельного пайплайна
# ==============================================================================

print("\nСобираем финальный деплоябельный пайплайн...")

final_deployable_pipeline = Pipeline([
    ('seq_to_smiles', SequenceToSmilesTransformer(sequence_col='standard_sequence', smiles_col='smiles_sequence')),
    ('cell_line_encoder', fitted_cell_line_encoder), # Используем уже обученный энкодер
    ('descriptor_calc', DescriptorCalculatorTransformer(
        sequence_col='standard_sequence', smiles_col='smiles_sequence')),
    ('imputer', fitted_imputer),        # Используем обученный SimpleImputer
    ('var_threshold', fitted_var_selector), # Используем обученный VarianceThresholdTransformer
    ('scaler', fitted_scaler),           # Используем обученный MinMaxScaler
    ('pca', fitted_pca),                 # Используем обученный PCA
    ('svr', fitted_svr_model)            # Используем обученную SVR модель
])

print("Финальный пайплайн собран. Тестируем его на сырых данных.")

# Для оценки всего пайплайна, передаем ему сырые данные (X_raw, включающие последовательность и cell_line)
# Важно: X_raw_for_full_pipeline_test должен иметь структуру: 'standard_sequence', 'cell_line'
# Мы должны создать X_raw_for_full_pipeline_test, чтобы он отражал формат, который будет принимать пайплайн
# на вход для предсказаний.
# Этот X_raw_for_full_pipeline_test должен содержать все строки из for_regr_descriptors_full.csv
# до любых фильтраций, но с корректными названиями колонок.

# Загружаем original_sequences_df (если у вас есть такой файл)
# Если у вас sequences.csv и X_cell_line_filtered.csv - это сырые данные для генерации дескрипторов
# тогда они должны быть объединены здесь
# В рамках этого примера, будем считать, что test_peptides.csv и test_cell_lines.csv
# представляют собой "новые" данные для предсказания.
# Но для оценки нам нужно использовать X_train_raw и X_test_raw из прошлого пайплайна
# который создавался из initial_peptides_df и cell_line_data_df.

# ИСХОДНЫЕ СЫРЫЕ ДАННЫЕ (как они были бы переданы в самом начале предыдущей версии пайплайна)
# Это данные, которые full_pipeline_for_deployment ожидает на вход
initial_peptides_df_raw = pd.read_csv('regr/datasets/sequences.csv')
cell_line_data_df_raw = pd.read_csv('regr/datasets/X_cell_line_filtered.csv')

# Применяем логику переименования из main, чтобы обеспечить 'standard_sequence'
if 'sequence' in initial_peptides_df_raw.columns and 'standard_sequence' not in initial_peptides_df_raw.columns:
    initial_peptides_df_raw.rename(columns={'sequence': 'standard_sequence'}, inplace=True)
elif 0 in initial_peptides_df_raw.columns and 'standard_sequence' not in initial_peptides_df_raw.columns:
    initial_peptides_df_raw.rename(columns={0: 'standard_sequence'}, inplace=True)
elif 'standard_sequence' not in initial_peptides_df_raw.columns:
    raise ValueError("The 'sequences.csv' file does not contain a 'standard_sequence' column "
                     "or a column that can be easily renamed to it. "
                     "Please check your CSV file's column names.")

# Объединяем сырые данные последовательностей и клеточных линий
initial_peptides_df_raw = initial_peptides_df_raw.reset_index(drop=True)
cell_line_data_df_raw = cell_line_data_df_raw.reset_index(drop=True)

if len(initial_peptides_df_raw) != len(cell_line_data_df_raw):
    raise ValueError("initial_peptides_df_raw and cell_line_data_df_raw must have the same number of rows to be concatenated by index.")

X_raw_for_full_pipeline_evaluation = pd.concat([initial_peptides_df_raw, cell_line_data_df_raw], axis=1)

# Убедимся, что колонка 'cell_line' имеет строковый тип для пайплайна
if 'cell_line' in X_raw_for_full_pipeline_evaluation.columns:
    X_raw_for_full_pipeline_evaluation['cell_line'] = X_raw_for_full_pipeline_evaluation['cell_line'].astype(str)

# Оцениваем модель на всех сырых данных (тренировочные + тестовые, как их видит пайплайн)
# Здесь нам нужно правильно сопоставить Y с X_raw_for_full_pipeline_evaluation
# Y для оценки должно быть тем же Y, которое использовалось для обучения
# (т.е. y_original_target, отфильтрованное и log1p-преобразованное)
# Это означает, что X_raw_for_full_pipeline_evaluation должен быть отфильтрован
# точно так же, как X_original_target.
# Поскольку full_deployable_pipeline не выполняет фильтрацию по uptake_type или выбросам,
# мы должны подать ему X_raw, которое *уже соответствует* y_original_target по количеству строк и порядку.

# Восстанавливаем X_raw, которые соответствуют y_original_target.
# Это df_filtered_outliers с добавленными колонками sequences.csv
X_train_final_raw_for_eval, X_test_final_raw_for_eval, y_train_final_for_eval, y_test_final_for_eval = train_test_split(
    X_raw_for_full_pipeline_evaluation.copy(),
    y_original_target,
    test_size=0.2, random_state=42
)


# Оцениваем весь пайплайн на тренировочных и тестовых данных
print("\n--- Final Deployable Pipeline Evaluation (on raw-like data) ---")
evaluate_model(final_deployable_pipeline, X_train_final_raw_for_eval, y_train_final_for_eval, 
               X_test_final_raw_for_eval, y_test_final_for_eval)

# Сохранение финального пайплайна
output_dir = 'trained_pipelines'
os.makedirs(output_dir, exist_ok=True)
pipeline_filename = os.path.join(output_dir, 'SVR_Full_Deployable_Pipeline.joblib')
joblib.dump(final_deployable_pipeline, pipeline_filename)
print(f"\nФинальный деплоябельный пайплайн успешно сохранен как '{pipeline_filename}'")