import pandas as pd
import joblib
import argparse  # Для удобного запуска из командной строки
from pathlib import Path 
import sys
from preprocessing_utils import (
    categorize_sequences,
    add_sequence_features,
    validate_sequences,
    add_descriptors_features
)

# Основные метрики на обучающем датасете:
#  - Accuracy (общая точность): 85.58%
#  - Precision (точность для класса 'CPP'): 0.9902
#    (Из всех, что модель назвала 'CPP', сколько на самом деле ими являются)
#  - Recall (полнота для класса 'CPP'): 0.7266
#    (Сколько реальных 'CPP' модель смогла найти)

#Полный отчет по классификации:
#              precision    recall  f1-score   support

# non-CPP (0)       0.77      0.99      0.87      1321
#     CPP (1)       0.99      0.73      0.84      1397

#    accuracy                           0.86      2718
#   macro avg       0.88      0.86      0.85      2718
# weighted avg       0.89      0.86      0.85      2718


# --- Глобальные переменные ---
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_PATH = SCRIPT_DIR / 'peptide_classifier_artifacts.joblib'

def load_artifacts(path: str) -> dict:
    """Загружает артефакты из файла joblib."""
    print(f"[*] Загрузка артефактов из файла: {path}")
    try:
        artifacts = joblib.load(path)
        print("    - Модель загружена")
        print("    - Порог загружен")
        print("    - LabelEncoder загружен")
        return artifacts
    except FileNotFoundError:
        print(f"[!] Ошибка: Файл артефактов не найден по пути: {path}")
        exit(1)

def full_preprocessing_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Полный пайплайн предобработки для новых, "сырых" данных.
    Эта функция должна в точности повторять шаги, сделанные перед обучением.
    """
    print("[*] Запуск полного пайплайна предобработки...")
    
    # Шаг 1: Генерация признаков из последовательностей
    processed_df = (raw_df.copy()
                    .pipe(categorize_sequences)
                    .pipe(add_sequence_features)
                    .pipe(validate_sequences)
                    .pipe(lambda df: df.dropna(subset=['standard_sequence']))
                    .pipe(add_descriptors_features))

    # Шаг 3: Конвертация булевых колонок в int
    bool_cols = processed_df.columns[processed_df.dtypes == 'bool'].tolist()
    for col in bool_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(int)
    
    print("[+] Предобработка завершена.")
    return processed_df

def predict(data: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Принимает предобработанные данные и артефакты, возвращает DataFrame с предсказаниями.
    """
    # Извлекаем компоненты из словаря
    model = artifacts['model']
    threshold = artifacts['threshold']
    label_encoder = artifacts['label_encoder']
    
    # Получаем названия признаков, на которых обучалась модель
    # (модель scikit-learn хранит их после обучения)
    feature_names = model.feature_names_in_
    
    # Убеждаемся, что в данных есть все нужные колонки и они в правильном порядке
    X_new = data[feature_names]
    
    print("[*] Получение предсказаний модели...")
    # 1. Получаем вероятности для класса "1"
    probas = model.predict_proba(X_new)[:, 1]
    
    # 2. Применяем порог для получения бинарных предсказаний (0 или 1)
    predictions_numeric = (probas >= threshold).astype(int)

    
    # Создаем DataFrame с результатами для наглядности
    results_df = pd.DataFrame({
        'sequence': data['sequence'],
        'probability_cpp': probas,
        'prediction_numeric': predictions_numeric,
    })
    
    print("[+] Предсказания готовы.")
    return results_df


if __name__ == '__main__':
    print("--- НАЧАЛО ПРОЦЕССА ОЦЕНКИ МОДЕЛИ ---")

    # 1. Загружаем артефакты модели
    loaded_artifacts = load_artifacts(ARTIFACTS_PATH)
    print(f"[*] Используемый порог отсечения: {loaded_artifacts['threshold']:.4f}")

    # 2. Читаем исходные данные из файла
    RAW_DATA_PATH = 'C:/Users/ALI/itmo-cpp/input_data/all_peptides_for_classification.csv'
    print(f"[*] Чтение данных из файла: {RAW_DATA_PATH}")
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"[!] Ошибка: Файл с данными не найден по пути: {RAW_DATA_PATH}")
        sys.exit(1)
        
    # 3. Преобразуем булевый столбец is_cpp в числовой формат (0/1) для сравнения
    if 'is_cpp' not in raw_df.columns:
        print(f"[!] Ошибка: В файле {RAW_DATA_PATH} отсутствует целевой столбец 'is_cpp'.")
        sys.exit(1)
    raw_df['is_cpp'] = raw_df['is_cpp'].astype(int)

    # 4. Прогоняем данные через пайплайн предобработки
    processed_data = full_preprocessing_pipeline(raw_df)

    # 5. Делаем предсказания. Получаем DataFrame с 'sequence' и 'prediction_numeric'
    prediction_results_df = predict(processed_data, loaded_artifacts)

    # 6. Объединяем исходные данные с предсказаниями по колонке 'sequence',
    # чтобы сопоставить истинные значения 'is_cpp' с 'prediction_numeric'.
    # Это самый надежный способ, даже если предобработка удалила некоторые строки.
    final_report_df = pd.merge(
        raw_df[['sequence', 'is_cpp']], # Берем только нужные колонки из исходных данных
        prediction_results_df,         # Берем все колонки из предсказаний
        on='sequence',                 # Ключ для объединения
        how='inner'                    # Оставляем только те строки, что есть в обоих датафреймах
    )

    # 7. Добавляем столбец для сравнения и считаем процент совпадений
    final_report_df['is_match'] = (final_report_df['is_cpp'] == final_report_df['prediction_numeric'])
    accuracy = final_report_df['is_match'].mean() * 100

    # 8. Выводим результаты
    print("\n--- РЕЗУЛЬТАТЫ ОЦЕНКИ ---")
    print(f"\nПроцент совпадений (Accuracy): {accuracy:.2f}%\n")
    
    # Создаем финальный DataFrame для чистого вывода
    output_df = final_report_df[['is_cpp', 'prediction_numeric', 'is_match']]
    
    print("Пример результатов (первые 15 строк):")
    print(output_df.head(15).to_string())
    
    print("\n--- ПРОЦЕСС ЗАВЕРШЕН ---")


    