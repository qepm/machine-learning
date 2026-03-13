import os
import warnings
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Папка с исходными данными
DATA_DIR = os.path.join(BASE_DIR, "data")

# Папка, куда будем сохранять модель
MODEL_DIR = os.path.join(BASE_DIR, "model")

SESSIONS_PKL = os.path.join(DATA_DIR, "ga_sessions.pkl")
HITS_PKL = os.path.join(DATA_DIR, "ga_hits.pkl")

# Это запасной вариант
SESSIONS_CSV = os.path.join(DATA_DIR, "ga_sessions.csv")
HITS_CSV = os.path.join(DATA_DIR, "ga_hits.csv")

# Пути до файлов, которые будут сохранены после обучения модели
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")
CAT_FEATURES_PATH = os.path.join(MODEL_DIR, "cat_features.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.txt")

# загрузка датафремйма
def load_dataframe(pkl_path: str, csv_path: str) -> pd.DataFrame:
    if os.path.exists(pkl_path):
        print(f"Loading: {pkl_path}")
        return pd.read_pickle(pkl_path)

    print(f"Loading: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


# ============================================================
# подготтовка таблицы Sessions, чистим и преобразуем таблицу визитов
# копируем датафрейм, далее удаляем признак device_model, приводим даты к datetime, создаём новые признаки времени
# заполняем пропуски в категориальных колонках, создаём признак is_returning_user
def prepare_sessions(df_sessions: pd.DataFrame) -> pd.DataFrame:
    df = df_sessions.copy()

    if "device_model" in df.columns:
        df = df.drop(columns=["device_model"])

    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")

    # для извлечения признаков собираем полный datetime из даты и времени визита
    df["visit_datetime"] = pd.to_datetime(
        df["visit_date"].astype(str) + " " + df["visit_time"].astype(str),
        errors="coerce"
    )

    # Приводим ID к строкам, чтобы модель не считала session_id числом
    df["session_id"] = df["session_id"].astype(str)
    df["client_id"] = df["client_id"].astype(str)

    # Список колонок, где пропуски заполним значением "unknown", чтобы не потерять эти строки
    fill_unknown_cols = [
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_adcontent",
        "utm_keyword",
        "device_category",
        "device_os",
        "device_brand",
        "device_browser",
        "geo_country",
        "geo_city",
        "device_screen_resolution",
    ]

    # Для каждой такой колонкизаменяем NaN на "unknown" и приводим к строке
    for col in fill_unknown_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    # Из visit_datetime извлекаем новые полезные признаки, в частонтсми, час визита
    df["visit_hour"] = df["visit_datetime"].dt.hour.fillna(-1).astype("int16")

    # Теперь день недели, где 0 = понедельник, а 6 = воскресенье
    df["visit_dayofweek"] = df["visit_datetime"].dt.dayofweek.fillna(-1).astype("int16")

    # Месяц визита
    df["visit_month"] = df["visit_datetime"].dt.month.fillna(-1).astype("int16")

    # Возвращался ли пользователь ранее, если visit_number > 1, значит это не первый визит
    df["is_returning_user"] = (df["visit_number"] > 1).astype("int8")

    # Почти весь трафик идёт из РФ,поэтому geo_country мало полезен
    if "geo_country" in df.columns:
        df = df.drop(columns=["geo_country"])

    return df


# подготовка таблицы HITS, из таблицы событий нам нужна только информация о том, был ли в сессии целевой event
# оставляем только session_id и event_action, создаём target_event = 1, если event_action == sub_submit_success
# агрегируем на уровень session_id, если хотя бы одно целевое событие было, ставим 1
def prepare_hits(df_hits: pd.DataFrame) -> pd.DataFrame:
    df = df_hits.copy()

    # Оставляем только нужные колонки
    keep_cols = ["session_id", "event_action"]
    df = df[keep_cols].copy()

    # session_id приводим к строке
    df["session_id"] = df["session_id"].astype(str)

    # Пропуски в event_action заменяем на "unknown"
    df["event_action"] = df["event_action"].fillna("unknown").astype(str)

    # Целевое событие sub_submit_success = успешная отправка формы
    target_events = ["sub_submit_success"]

    # Создаём бинарный target: 1 если событие целевое, иначе 0
    df["target_event"] = df["event_action"].isin(target_events).astype("int8")

    # Группируем по session_id и берём максимум: если в рамках одной сессии target_event хоть раз был равен 1,
    # значит вся сессия считается целевой
    target = (
        df.groupby("session_id", as_index=False)["target_event"]
        .max()
    )

    return target

# сборка финального датасета, в котором соединяем sessions и target по session_id
# Используем left join- хотим сохранить ВСЕ сессии - если в hits не нашлось целевого события, target будет NaN
# - потом заменим NaN на 0
def build_dataset(df_sessions: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    df = df_sessions.merge(target, on="session_id", how="left")

    # Если у сессии не было target-события, ставим 0
    df["target_event"] = df["target_event"].fillna(0).astype("int8")

    return df

# редактирование категорний - в этих признаках могут быть тысячи уникальных значений, поэтому редкие
# значения заменяем на "other".  редкость считаем только на train, чтобы не допустить data leakage
def reduce_cardinality(df: pd.DataFrame, train_index: pd.Index) -> pd.DataFrame:
    df = df.copy()

    cat_cols = [
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_adcontent",
        "utm_keyword",
        "device_category",
        "device_os",
        "device_brand",
        "device_browser",
        "geo_city",
        "device_screen_resolution",
    ]

    for col in cat_cols:
        if col not in df.columns:
            continue

        # Считаем частоты только на train
        vc = df.loc[train_index, col].value_counts(dropna=False)

        # Все категории, которые встречаются меньше 100 раз,считаем редкими
        rare_values = vc[vc < 100].index

        # Заменяем редкие категории на "other"
        df[col] = df[col].where(~df[col].isin(rare_values), "other")

    return df

# сохранение метрик в отдельный файл, чтобы потом можно было посмотреть, какое качество было у модели при обучении
def save_metrics(roc_auc: float, y_test: pd.Series, pred_labels: np.ndarray) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    report = classification_report(y_test, pred_labels, digits=4)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"ROC_AUC: {roc_auc:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    print(f"Metrics saved to: {METRICS_PATH}")

# функция обучения, где собран весь pipeline: загрузка данных,  подготовка,  сборка датасета, выбор признаков
# train/test split, снижение вариантов категорий , обучение CatBoost,  оценка модели,  сохранение
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Загружаем данные
    df_sessions = load_dataframe(SESSIONS_PKL, SESSIONS_CSV)
    df_hits = load_dataframe(HITS_PKL, HITS_CSV)

    print("Sessions shape:", df_sessions.shape)
    print("Hits shape:", df_hits.shape)

    # Готовим sessions и hits
    df_sessions = prepare_sessions(df_sessions)
    target = prepare_hits(df_hits)

    print("Prepared sessions shape:", df_sessions.shape)
    print("Prepared target shape:", target.shape)

    # Строим финальный датасет
    df = build_dataset(df_sessions, target)

    print("Final dataset shape:", df.shape)
    print("Target mean:", df["target_event"].mean())

    # Выбираем признаки для модели- еречисляем только те признаки,которые хотим использовать в обучении
    feature_cols = [
        "visit_number",
        "visit_hour",
        "visit_dayofweek",
        "visit_month",
        "is_returning_user",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_adcontent",
        "utm_keyword",
        "device_category",
        "device_os",
        "device_brand",
        "device_browser",
        "geo_city",
        "device_screen_resolution",
    ]

    # На случай, если какой-то признак отсутствует
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].copy()
    y = df["target_event"].copy()

    # Делим на train и test, stratify=y чтобы в train и test сохранилась примерно одинаковая доля
    # положительного класса
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Урезаем редкие категории-сначала объединяем train и test обратно,чтобы одной функцией обработать все строки
    X_all = pd.concat([X_train, X_test], axis=0)

    # Редкие категории считаем только на train_index
    X_all = reduce_cardinality(X_all, train_index=X_train.index)

    # Далее снова разделяем на train/test по индексам
    X_train = X_all.loc[X_train.index].copy()
    X_test = X_all.loc[X_test.index].copy()

    # Находим категориальные признаки. CatBoost умеет работать с ними напрямую, без OneHotEncoding, что удобно
    cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Создаём модель. CatBoost хорошо работает с категориальными данными, устойчив к пропускам или шуму
    model = CatBoostClassifier(
        iterations=400,              # максимальное число итераций
        depth=6,                     # глубина деревьев
        learning_rate=0.05,          # шаг обучения
        loss_function="Logloss",     # функция потерь для бинарной классификации
        eval_metric="AUC",           # основная метрика качества
        random_seed=42,              # фиксируем seed для воспроизводимости
        verbose=50,                  # печать каждые 50 итераций
        auto_class_weights="Balanced"  # автоматически балансируем классы
    )

    # Обучаем модель. eval_set чтобы модель отслеживала качество на test и могла выбрать лучшую итерацию
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # Считаем метрику ROC-AUC. predict_proba -> вероятность класса 1
    pred_proba = model.predict_proba(X_test)[:, 1]

    # ROC-AUC — главная метрика в этой задаче
    roc_auc = roc_auc_score(y_test, pred_proba)

    # Для classification_report строим бинарное решение по порогу 0.5
    pred_labels = (pred_proba >= 0.5).astype(int)

    print(f"\nROC-AUC: {roc_auc:.6f}")

    # Важность признаков. помогает понять, какие признаки сильнее всего влияют на прогноз
    fi = pd.Series(
        model.get_feature_importance(),
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\nTop-15 feature importance:")
    print(fi.head(15))

    # Сохраняем модель.
    # model.pkl           -> сама модель
    # feature_cols.pkl    -> список признаков
    # cat_features.pkl    -> список категориальных признаков
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    joblib.dump(cat_features, CAT_FEATURES_PATH)

    # Сохраняем метрики в текстовый файл
    save_metrics(roc_auc, y_test, pred_labels)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Features saved to: {FEATURES_PATH}")
    print(f"Categorical features saved to: {CAT_FEATURES_PATH}")


# точка входа. если файл запущен как самостоятельный скрипт, то выполняем main()
# можно запускать так: python src/train_model.py
if __name__ == "__main__":
    main()