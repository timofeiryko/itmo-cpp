import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.base import clone
from sklearn.cluster import KMeans
import joblib

# -----------------------------
# Конфигурация
# -----------------------------
@dataclass
class Config:
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2 
    target_precision: float = 0.95
    min_positive_preds_on_val: int = 5  
    budgets: List[int] = None 
    oof_cv: int = 5
    oof_n_estimators: int = 400 
    calibr_cv: int = 5
    calibr_method: str = 'isotonic'

    def __post_init__(self):
        if self.budgets is None:
            # Стартуем с 200 и наращиваем
            self.budgets = [200, 300, 400, 500, 600, 800, 1000]

# -----------------------------
# Вспомогательные функции
# -----------------------------

def compute_oof_probas(X, y, base_rf_params: Dict, cv=5, n_estimators_override: Optional[int]=None, random_state=42) -> np.ndarray:
    """OOF-оценки вероятности положительного класса для train pool."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        rf_params = base_rf_params.copy()
        if n_estimators_override is not None:
            rf_params['n_estimators'] = n_estimators_override
        clf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=random_state + fold)
        clf.fit(X[tr_idx], y[tr_idx])
        oof_proba[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    return oof_proba

def compute_embedding_for_diversity(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Стандартизованный эмбеддинг для диверсификации (евклид)."""
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    return Z, scaler

def k_center_greedy(embedding: np.ndarray, candidate_idx: np.ndarray, k: int, random_state: int = 42) -> List[int]:
    """Farthest-first k-center отбор индексов из candidate_idx по embedding (евклид)."""
    if k <= 0 or len(candidate_idx) == 0:
        return []
    if len(candidate_idx) <= k:
        return candidate_idx.tolist()

    rng = np.random.RandomState(random_state)
    C = candidate_idx
    E = embedding[C]

    # Первый центр: самый удалённый от среднего (устойчивее, чем случайный)
    centroid = E.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(E - centroid, axis=1)
    first = np.argmax(dists)
    selected = [C[first]]

    # Инициализация расстояний до ближайшего центра
    min_dist = pairwise_distances(E, E[[first]], metric='euclidean').reshape(-1)

    while len(selected) < k:
        next_idx = np.argmax(min_dist)
        selected.append(C[next_idx])
        # обновляем min_dist
        new_d = pairwise_distances(E, E[[next_idx]], metric='euclidean').reshape(-1)
        min_dist = np.minimum(min_dist, new_d)

    return selected

def find_threshold_for_precision_max_f1(y_true, proba, target_precision, min_pos=1):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # thresholds имеет длину len(precision)-1; согласуем индексы
    valid = np.where(precision[:-1] >= target_precision)[0]
    if len(valid) == 0:
        return None, {'precision': None, 'recall': None, 'f1': None, 'ap': average_precision_score(y_true, proba)}
    # выбираем порог с макс. F1 среди допустимых
    best_idx, best_f1 = None, -1.0
    for i in valid:
        thr = thresholds[i]
        y_pred = (proba >= thr).astype(int)
        if y_pred.sum() < min_pos:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_idx = f1, i
    if best_idx is None:
        return None, {'precision': None, 'recall': None, 'f1': None, 'ap': average_precision_score(y_true, proba)}
    thr = thresholds[best_idx]
    y_pred = (proba >= thr).astype(int)
    return float(thr), {
        'precision': precision[best_idx],
        'recall': recall[best_idx],
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'ap': average_precision_score(y_true, proba)
    }

def select_subset_indices_v3(
    X_pool, y_pool, oof_proba, B, embedding,
    core_set_ratio=0.7, # 70% бюджета на "фундамент"
    hard_neg_ratio=0.2, # 20% на "сложные негативные"
    margin_ratio=0.1,   # 10% на "пограничные"
    n_clusters_per_class=10, # Количество прототипов для каждого класса
    hard_neg_top_quantile=0.25,
    random_state=42
):
    n = len(y_pool)
    B_core = int(round(B * core_set_ratio))
    B_hneg = int(round(B * hard_neg_ratio))
    B_margin = max(0, B - B_core - B_hneg)

    selected = []
    
    # --- ЭТАП 1: ФУНДАМЕНТ (Core-Set) ---
    # Делим бюджет на "фундамент" пропорционально балансу классов
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]
    
    pos_ratio = len(pos_idx) / n
    B_core_pos = int(round(B_core * pos_ratio))
    B_core_neg = max(0, B_core - B_core_pos)

    # Находим прототипы для позитивного класса
    if B_core_pos > 0 and len(pos_idx) > n_clusters_per_class:
        kmeans_pos = KMeans(n_clusters=n_clusters_per_class, random_state=random_state, n_init='auto')
        clusters = kmeans_pos.fit_predict(embedding[pos_idx])
        distances = kmeans_pos.transform(embedding[pos_idx])
        
        # Находим по 1 самому близкому к центру в каждом кластере
        core_pos_indices = []
        for i in range(n_clusters_per_class):
            cluster_members = np.where(clusters == i)[0]
            if len(cluster_members) > 0:
                closest_point_idx = cluster_members[np.argmin(distances[cluster_members, i])]
                core_pos_indices.append(pos_idx[closest_point_idx])
        
        # Если прототипов мало, добираем самых разнообразных из оставшихся
        sel_core_pos = k_center_greedy(embedding, np.array(list(set(core_pos_indices))), B_core_pos, random_state)
        selected.extend(sel_core_pos)

    # Находим прототипы для негативного класса (аналогично)
    if B_core_neg > 0 and len(neg_idx) > n_clusters_per_class:
        kmeans_neg = KMeans(n_clusters=n_clusters_per_class, random_state=random_state+1, n_init='auto')
        clusters = kmeans_neg.fit_predict(embedding[neg_idx])
        distances = kmeans_neg.transform(embedding[neg_idx])
        
        core_neg_indices = []
        for i in range(n_clusters_per_class):
            cluster_members = np.where(clusters == i)[0]
            if len(cluster_members) > 0:
                closest_point_idx = cluster_members[np.argmin(distances[cluster_members, i])]
                core_neg_indices.append(neg_idx[closest_point_idx])

        sel_core_neg = k_center_greedy(embedding, np.array(list(set(core_neg_indices))), B_core_neg, random_state+1)
        selected.extend(sel_core_neg)

    # --- ЭТАП 2: УТОЧНЕНИЕ ---
    already_selected = np.array(list(set(selected)), dtype=int)
    
    # Hard-негативы
    if B_hneg > 0:
        neg_sorted_desc = neg_idx[np.argsort(-oof_proba[neg_idx])]
        k_hn = max(1, int(len(neg_idx) * hard_neg_top_quantile))
        hn_candidates = np.setdiff1d(neg_sorted_desc[:k_hn], already_selected, assume_unique=False)
        sel_hneg = k_center_greedy(embedding, hn_candidates, min(B_hneg, len(hn_candidates)), random_state=random_state+2)
        selected.extend(sel_hneg)
    
    # Margin
    already_selected = np.array(list(set(selected)), dtype=int)
    if B_margin > 0:
        margin = np.abs(oof_proba - 0.5)
        order = np.argsort(margin)
        k_cand = min(len(order), max(B_margin * 5, B_margin)) # берем больше кандидатов
        marg_candidates = np.setdiff1d(order[:k_cand], already_selected, assume_unique=False)
        sel_marg = k_center_greedy(embedding, marg_candidates, min(B_margin, len(marg_candidates)), random_state=random_state+3)
        selected.extend(sel_marg)

    # Финальная уникализация и дозаполнение, если нужно
    final_selected = list(dict.fromkeys(selected))
    if len(final_selected) < B:
        remaining = np.setdiff1d(np.arange(n), np.array(final_selected, dtype=int))
        extra = k_center_greedy(embedding, remaining, B - len(final_selected), random_state=random_state+4)
        final_selected.extend(extra)
        
    return np.array(final_selected[:B], dtype=int)

def fit_calibrated_rf(X_train: np.ndarray, y_train: np.ndarray, rf_params: Dict, cv: int = 5, method: str = 'isotonic', random_state: int = 42):
    rf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=random_state)
    clf = CalibratedClassifierCV(estimator=rf, method=method, cv=cv, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def save_classifier_artifacts(model, threshold, feature_names, path, meta=None):
    """
    Сохраняет артефакты классификатора:
      - model: CalibratedClassifierCV (или любой sklearn-классификатор)
      - threshold: float (порог для бинаризации вероятностей)
      - feature_names: список/np.array имен фичей, чтобы инференс мог взять нужные колонки
      - meta: опциональные метаданные (для трассировки)
    """
    # Гарантируем наличие feature_names_in_ (инференс это использует)
    if not hasattr(model, "feature_names_in_"):
        model.feature_names_in_ = np.array(feature_names, dtype=object)

    artifacts = {
        "model": model,
        "threshold": float(threshold),
    }
    if meta is not None:
        artifacts["meta"] = meta

    joblib.dump(artifacts, path)
    print(f"Artifacts saved to: {path}")

def stratified_train_val_test_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает индексы пул/валидация/тест при стратифицированном разбиении.
    Это даёт «глобальные» индексы относительно исходного df_ready.
    """
    idx_all = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_size, stratify=y, random_state=random_state
    )
    y_trainval = y[idx_trainval]
    val_rel_size = val_size / (1.0 - test_size)
    idx_pool, idx_val = train_test_split(
        idx_trainval, test_size=val_rel_size, stratify=y_trainval, random_state=random_state
    )
    return idx_pool, idx_val, idx_test

# -----------------------------
# Главная процедура пайплайна
# -----------------------------
def run_active_sampling_pipeline_2(
    X: np.ndarray,
    y: np.ndarray,
    rf_params_final: Dict,
    config: Config = Config(),
    feature_names: Optional[List[str]] = None,
    artifacts_path: Optional[str] = None
):
    rs = config.random_state
    # 0) Сплиты по индексам → затем получаем массивы
    idx_pool, idx_val, idx_test = stratified_train_val_test_indices(
        y, test_size=config.test_size, val_size=config.val_size, random_state=rs
    )
    X_pool, y_pool = X[idx_pool], y[idx_pool]
    X_val,  y_val  = X[idx_val],  y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    print(f"Shapes: pool={X_pool.shape}, val={X_val.shape}, test={X_test.shape}")

    # 1) OOF-прогнозы на pool (учитель RF с облегчённым числом деревьев)
    base_rf_params_for_oof = rf_params_final.copy()
    oof_proba = compute_oof_probas(
        X_pool, y_pool,
        base_rf_params_for_oof,
        cv=config.oof_cv,
        n_estimators_override=config.oof_n_estimators,
        random_state=rs
    )
    # 2) Эмбеддинг для диверсификации (евклид на стандартизованных фичах)
    embedding, scaler = compute_embedding_for_diversity(X_pool)

    results = []
    best_solution = None

    for B in config.budgets:
        # 2a) Отбор индексов поднабора размером B
        sel_idx = select_subset_indices_v3(
            X_pool, y_pool, oof_proba, B, embedding,
            core_set_ratio=0.7, 
            hard_neg_ratio=0.2,
            margin_ratio=0.1,  
            n_clusters_per_class=10,
            hard_neg_top_quantile=0.25,
            random_state=rs
        )
        X_sub, y_sub = X_pool[sel_idx], y_pool[sel_idx]

        # 3) Обучение откалиброванной финальной модели на поднаборе
        clf = fit_calibrated_rf(
            X_sub, y_sub, rf_params_final, cv=config.calibr_cv,
            method=config.calibr_method, random_state=rs
        )

        # 4) Порог под целевую precision — по валидации
        proba_val = clf.predict_proba(X_val)[:, 1]
        thr, val_metrics = find_threshold_for_precision_max_f1(
            y_val, proba_val, config.target_precision, min_pos=config.min_positive_preds_on_val
        )

        if thr is not None:
            # оценка на тесте (не используем для выбора, только логируем)
            proba_test = clf.predict_proba(X_test)[:, 1]
            y_pred_val = (proba_val >= thr).astype(int)
            y_pred_test = (proba_test >= thr).astype(int)
            test_prec = precision_score(y_test, y_pred_test, zero_division=0)
            test_rec = recall_score(y_test, y_pred_test, zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

            result = {
                'B': B,
                'threshold': thr,
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_AP': val_metrics['ap'],
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_f1': test_f1,
                'selected_indices': sel_idx,
                'selected_global_indices': idx_pool[sel_idx].tolist()
            }
            results.append(result)

            print(f"[B={B}] VAL: P={val_metrics['precision']:.3f}, R={val_metrics['recall']:.3f}, F1={val_metrics['f1']:.3f}, thr={thr:.4f} | "
                  f"TEST: P={test_prec:.3f}, R={test_rec:.3f}, F1={test_f1:.3f}")

            if artifacts_path is not None:
                # Внимание: clf обучен на np.ndarray → добавим feature_names явно
                if feature_names is None:
                    # Если по какой-то причине список фичей не передали, попытаемся
                    # взять из модели; если и там пусто — поднимем понятную ошибку
                    if not hasattr(clf, "feature_names_in_"):
                        raise ValueError("feature_names для сохранения не заданы и отсутствуют в модели. "
                                        "Передайте feature_names в run_active_sampling_pipeline_2.")
                    fnames = list(clf.feature_names_in_)
                else:
                    fnames = list(feature_names)

                meta = {
                    "precision_target": config.target_precision,
                    "calibration": config.calibr_method,
                    "B": B,
                    "selector": "v3 core-set (kMeans)",
                    "val": {"precision": result['val_precision'], "recall": result['val_recall'],
                            "f1": result['val_f1'], "AP": result['val_AP']},
                    "test": {"precision": result['test_precision'], "recall": result['test_recall'],
                            "f1": result['test_f1']},
                    "selected_count": len(sel_idx),
                }
                # Сохраняем
                save_classifier_artifacts(
                    model=clf,
                    threshold=thr,
                    feature_names=fnames,
                    path=artifacts_path,
                    meta=meta
                )

            # Ранняя остановка: достигли целевой precision на валидации
            if best_solution is None:
                best_solution = result
                print(f"--> Стоп-критерий выполнен на валидации при B={B}. Фиксируем минимальный поднабор.")
                break
        else:
            print(f"[B={B}] Не найден порог для precision >= {config.target_precision:.2f} (или слишком мало положительных предсказаний). Увеличиваем бюджет.")

    if best_solution is None and len(results) > 0:
        # если ни разу не выполнен критерий, берём лучшее по вал. precision
        best_solution = max(results, key=lambda r: r['val_precision'] if r['val_precision'] is not None else -1)
        print(f"Внимание: целевой precision не достигнут на валидации, выбран лучший по precision: B={best_solution['B']}.")

    return {
        'best': best_solution,
        'all_results': results,
        'test_global_indices': idx_test.tolist()
    }

# -----------------------------
# Пример использования
# -----------------------------
if __name__ == "__main__":
    # Предположим, у вас уже есть X (np.ndarray, shape [n_samples, 18]) и y (np.ndarray, shape [n_samples], 0/1)
    # X, y = ...  # загрузите ваши данные

    # Ваши гиперпараметры RF
    rf_params_final = dict(
        n_estimators=1200,
        max_depth=30,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features='log2',
        bootstrap=False,
        class_weight='balanced_subsample'
    )

    cfg = Config(
        random_state=42,
        test_size=0.2,
        val_size=0.2,
        target_precision=0.90,
        min_positive_preds_on_val=5,
        budgets=[500, 600, 800, 1000],
        oof_cv=5,
        oof_n_estimators=400,   # быстрее, чем 1200, для OOF-оценок
        calibr_cv=5,
        calibr_method='isotonic',
    )

    # Вызов пайплайна:
    df_ready = pd.read_csv("input_data/clean_peptides_for_classification_descriptors_with_id.csv")
    columns_to_use = [
    'seq_length', 'molecular_weight', 'nh3_tail', 'po3_pos',
    'biotinylated', 'acylated_n_terminal', 'cyclic', 'amidated',
    'stearyl_uptake', 'hexahistidine_tagged', 'aromaticity',
    'instability_index', 'isoelectric_point', 'helix_fraction',
    'turn_fraction', 'sheet_fraction', 'molar_extinction_coefficient_reduced',
    'molar_extinction_coefficient_oxidized', 'gravy'
    ]
    artifacts_path = "classifier_artifacts_active_learning.joblib"
    X = df_ready[columns_to_use].to_numpy()
    y = df_ready['is_cpp'].to_numpy()
    results = run_active_sampling_pipeline_2(X, y, rf_params_final, cfg,
        feature_names=columns_to_use, artifacts_path=artifacts_path)
    print("Лучшее решение:", results['best'])

    best = results['best']
    if best and 'selected_global_indices' in best:
        # 1) Получаем id обучающих строк из df_ready по глобальным индексам
        sel_global = np.unique(np.array(best['selected_global_indices'], dtype=int))
        # Проверим, что id уникальный и присутствует
        if 'id' not in df_ready.columns:
            raise KeyError("В df_ready нет столбца 'id'. Невозможно сопоставить строки.")
        assert df_ready['id'].is_unique, "Ожидается уникальный столбец 'id' в df_ready."
        train_ids = df_ready.iloc[sel_global]['id'].values

        # 2) Загружаем исходный (непредобработанный) датасет all.csv
        all_path = "input_data/all_peptides_for_classification.csv"  # при необходимости поменяйте путь
        all_df = pd.read_csv(all_path)
        if 'id' not in all_df.columns:
            raise KeyError(f"В {all_path} нет столбца 'id'. Невозможно удалить строки по id.")
        assert all_df['id'].is_unique, f"Ожидается уникальный столбец 'id' в {all_path}."

        # 3) Выравниваем типы id и строим маску удаления
        #    (важно, чтобы типы совпадали: строки/числа)
        train_ids = train_ids.astype(all_df['id'].dtype, copy=False)

        before = len(all_df)
        cleaned_df = all_df.loc[~all_df['id'].isin(train_ids)].copy()
        removed = before - len(cleaned_df)
        print(f"Будет удалено строк: {removed} из {before} (по id).")

        # 4) Сохраняем очищенный датасет
        out_path = "input_data/peptides_without_train_rows.csv"  # итоговый файл
        # cleaned_df.to_csv(out_path, index=False)
        print(f"Сохранён очищенный датасет: {out_path} | shape={cleaned_df.shape}")

        # Сохранить и сами обучающие строки из all.csv для аудита
        train_rows_path = "input_data/train_rows_for_active_sampling.csv"
        # all_df[all_df['id'].isin(train_ids)].to_csv(train_rows_path, index=False)
        print(f"Сохранены обучающие строки: {train_rows_path}")

        # Получаем индексы тестовой выборки
        if 'test_global_indices' in results:
            test_global = np.unique(np.array(results['test_global_indices'], dtype=int))
            
            test_ids = df_ready.iloc[test_global]['id'].values
            test_ids = test_ids.astype(all_df['id'].dtype, copy=False)
            test_df = all_df[all_df['id'].isin(test_ids)].copy()
            
            # Сохраняем в отдельный CSV
            test_rows_path = "input_data/test_rows_for_active_sampling.csv"
            test_df.to_csv(test_rows_path, index=False)
            print(f"Сохранены тестовые строки: {test_rows_path} | shape={test_df.shape}")
        else:
            print("Warning: Индексы тестовой выборки не найдены в результатах.")

    else:
        print("Warning: can't remove train rows from all.csv (no selected_global_indices in results['best']).")
    pass