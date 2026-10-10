"""Entrenamiento y evaluación de modelos para predecir la calidad del vino.

Migración del notebook TC5061_Notebook_Baseline_RuizJessica.ipynb.
El notebook se conserva solo para el EDA; todo lo demás vive aquí.

Uso (todos los argumentos son opcionales):
    python train.py
    python train.py --n-estimators 500 --max-depth 8 --min-samples-leaf 3
    python train.py --random-state 7 --test-size 0.25
    python train.py --tracking-uri http://localhost:5000
    python train.py --no-tune --no-mlflow
"""

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "quality"
ID_COL = "Id"
N_SPLITS = 5

# MLflow 3.x guarda los modelos de sklearn con skops, que pide declarar los
# tipos de confianza. Este es el nodo interno de los árboles (modelos propios).
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Valores por defecto (los mismos que usaba el notebook)
DEFAULT_DATA_PATH = "wine_quality_df.csv"
DEFAULT_EXPERIMENT = "wine_quality"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH = None
DEFAULT_MIN_SAMPLES_LEAF = 1

PARAM_GRID = {
    "max_depth": [5, 8, 12, None],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": ["sqrt", 0.5, 1.0],
}


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Fija las semillas de Python y NumPy.

    Además, la MISMA semilla se pasa de forma explícita a:
      - train_test_split (división entrenamiento/prueba),
      - RandomForestRegressor (en ambas versiones, sin ajustar y ajustada),
      - KFold (validación cruzada y GridSearchCV).
    Con los mismos argumentos y las mismas versiones de librerías
    (ver requirements.txt) dos ejecuciones producen el mismo resultado.
    """
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Lee el CSV y usa `Id` como índice (no es una variable del vino)."""
    return pd.read_csv(path, index_col=ID_COL)


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados (ignorando el Id) y reinicia el índice.

    Los outliers se conservan: son valores plausibles en vinos reales.
    """
    return df.drop_duplicates().reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    """Separa las variables fisicoquímicas (X) de la calidad (y)."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return X, y


def split_data(X, y, test_size=DEFAULT_TEST_SIZE,
               random_state=DEFAULT_RANDOM_STATE):
    """División entrenamiento/prueba estratificada por calidad."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def train_dummy(X_train, y_train) -> DummyRegressor:
    """Referencia: predice siempre la media de entrenamiento."""
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    return dummy


def train_linear_regression(X_train, y_train):
    """Modelo base: regresión lineal con variables estandarizadas."""
    modelo = make_pipeline(StandardScaler(), LinearRegression())
    modelo.fit(X_train, y_train)
    return modelo


def train_random_forest(X_train, y_train,
                        n_estimators=DEFAULT_N_ESTIMATORS,
                        max_depth=DEFAULT_MAX_DEPTH,
                        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
                        random_state=DEFAULT_RANDOM_STATE):
    """Random Forest sin ajustar, con hiperparámetros configurables."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def tune_random_forest(X_train, y_train, cv,
                       n_estimators=DEFAULT_N_ESTIMATORS,
                       random_state=DEFAULT_RANDOM_STATE):
    """Busca hiperparámetros con GridSearchCV usando SOLO entrenamiento.

    Devuelve (mejor modelo, mejores hiperparámetros).
    """
    grid = GridSearchCV(
        RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ),
        PARAM_GRID,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def compute_metrics(model, X, y) -> dict:
    """RMSE, MAE y R² de un modelo sobre un conjunto de datos."""
    pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def evaluate_models(models, X_train, y_train, X_test, y_test, X, y, cv):
    """Tabla comparativa con métricas de prueba, R² de entrenamiento y R² en
    validación cruzada (sobre todos los datos limpios)."""
    filas = []
    for nombre, modelo in models.items():
        test = compute_metrics(modelo, X_test, y_test)
        train = compute_metrics(modelo, X_train, y_train)
        cv_r2 = cross_validate(modelo, X, y, cv=cv, scoring="r2")["test_score"]
        filas.append(
            {
                "Modelo": nombre,
                "RMSE (prueba)": test["rmse"],
                "MAE (prueba)": test["mae"],
                "R² (prueba)": test["r2"],
                "R² (entrenamiento)": train["r2"],
                "R² (validación cruzada)": cv_r2.mean(),
            }
        )
    return pd.DataFrame(filas).set_index("Modelo").round(3)


def get_feature_importances(rf, columns) -> pd.Series:
    """Importancia de variables de un Random Forest, de menor a mayor."""
    return pd.Series(rf.feature_importances_, index=columns).sort_values()


# ---------------------------------------------------------------------------
# MLflow Tracking
# ---------------------------------------------------------------------------
def _save_metrics_file(metricas: dict, path: Path):
    """Guarda las métricas de la corrida en un JSON (artefacto)."""
    path.write_text(json.dumps(metricas, indent=2), encoding="utf-8")


def _save_pred_plot(modelo, X_test, y_test, path: Path):
    """Gráfica de calidad real vs. predicha en prueba (artefacto)."""
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: el script corre en terminal
    import matplotlib.pyplot as plt

    pred = modelo.predict(X_test)
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(y_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(np.asarray(y_test) + jitter, pred, alpha=0.5)
    ax.plot([3, 8], [3, 8], "r--", label="Predicción perfecta")
    ax.set_xlabel("quality real")
    ax.set_ylabel("quality predicha")
    ax.set_title("Real vs predicho (prueba)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_model_table(modelo, columnas, path: Path):
    """Coeficientes (regresión lineal) o importancias (Random Forest)."""
    if hasattr(modelo, "feature_importances_"):
        serie = pd.Series(modelo.feature_importances_, index=columnas)
        nombre = "importancia"
    elif hasattr(modelo, "steps") and hasattr(modelo[-1], "coef_"):
        serie = pd.Series(modelo[-1].coef_, index=columnas)
        nombre = "coeficiente"
    else:
        return False
    serie.sort_values().to_csv(path, header=[nombre])
    return True


def log_run(nombre, modelo, parametros, splits, sufijo=""):
    """Registra UNA corrida: parámetros, métricas, artefactos y modelo."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    X_train, X_test, y_train, y_test = splits

    run_name = f"{nombre} | {sufijo}" if sufijo else nombre
    with mlflow.start_run(run_name=run_name) as run:
        # 1) Hiperparámetros y configuración de la corrida
        for clave, valor in parametros.items():
            mlflow.log_param(clave, valor)

        # 2) Métricas de evaluación (entrenamiento y prueba)
        metricas = {}
        for conjunto, (A, b) in {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
        }.items():
            for metrica, valor in compute_metrics(modelo, A, b).items():
                metricas[f"{conjunto}_{metrica}"] = valor
                mlflow.log_metric(f"{conjunto}_{metrica}", valor)

        # 3) Artefactos: archivo de métricas, gráfica y tabla del modelo
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _save_metrics_file(metricas, tmp / "metricas.json")
            mlflow.log_artifact(str(tmp / "metricas.json"))

            _save_pred_plot(modelo, X_test, y_test, tmp / "real_vs_predicho.png")
            mlflow.log_artifact(str(tmp / "real_vs_predicho.png"))

            if _save_model_table(modelo, X_train.columns, tmp / "variables.csv"):
                mlflow.log_artifact(str(tmp / "variables.csv"))

        # 4) Modelo entrenado
        signature = infer_signature(X_train, modelo.predict(X_train))
        mlflow.sklearn.log_model(
            modelo, "modelo", signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        return run.info.run_id


def log_to_mlflow(models, hiperparametros, splits, parametros_comunes,
                  experiment=DEFAULT_EXPERIMENT, tracking_uri=None, sufijo=""):
    """Crea el experimento y registra una corrida por cada modelo."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    for nombre, modelo in models.items():
        parametros = {
            "modelo": type(modelo).__name__,
            **parametros_comunes,
            **{f"hiper_{k}": v
               for k, v in hiperparametros.get(nombre, {}).items()},
        }
        run_id = log_run(nombre, modelo, parametros, splits, sufijo)
        print(f"  [{nombre}] run_id: {run_id}")


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def main(args):
    set_global_seed(args.random_state)
    df = load_data(args.data_path)
    df_clean = clean_data(df)
    n_duplicados = len(df) - len(df_clean)
    print(f"Filas originales: {len(df)} | tras limpiar: {len(df_clean)} "
          f"({n_duplicados} duplicados eliminados)")

    X, y = split_features_target(df_clean)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Entrenamiento: {X_train.shape} | Prueba: {X_test.shape}")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=args.random_state)

    rf_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
    }
    models = {
        "Referencia (media)": train_dummy(X_train, y_train),
        "Regresión lineal": train_linear_regression(X_train, y_train),
        "Random Forest": train_random_forest(
            X_train, y_train, random_state=args.random_state, **rf_params
        ),
    }
    hiperparametros = {"Random Forest": rf_params}

    if not args.no_tune:
        rf_tuned, best_params = tune_random_forest(
            X_train, y_train, cv,
            n_estimators=args.n_estimators, random_state=args.random_state,
        )
        models["Random Forest ajustado"] = rf_tuned
        hiperparametros["Random Forest ajustado"] = best_params
        print("Mejores hiperparámetros:", best_params)

    comparacion = evaluate_models(
        models, X_train, y_train, X_test, y_test, X, y, cv
    )
    print("\n", comparacion.to_string(), sep="")

    mejor_rf = models.get("Random Forest ajustado", models["Random Forest"])
    print("\nImportancia de variables (Random Forest):")
    print(get_feature_importances(mejor_rf, X.columns).to_string())

    if not args.no_mlflow:
        print("\nRegistrando corridas en MLflow...")
        parametros_comunes = {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratify": TARGET,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_variables": X_train.shape[1],
            "duplicados_eliminados": n_duplicados,
            "outliers": "conservados",
        }
        # Sufijo que distingue las corridas en la UI según la configuración
        sufijo = (f"n{args.n_estimators}-d{args.max_depth}-"
                  f"leaf{args.min_samples_leaf}-seed{args.random_state}")
        log_to_mlflow(
            models, hiperparametros,
            (X_train, X_test, y_train, y_test), parametros_comunes,
            experiment=args.experiment_name, tracking_uri=args.tracking_uri,
            sufijo=sufijo,
        )
        print(f"Listo (experimento '{args.experiment_name}').")

    return comparacion


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa modelos de calidad de vino.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Datos y ejecución
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                        help="Ruta al CSV del dataset")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                        help="Proporción del conjunto de prueba")
    parser.add_argument("--random-state", type=int,
                        default=DEFAULT_RANDOM_STATE,
                        help="Semilla aleatoria (división, modelos y CV)")
    # Hiperparámetros del Random Forest
    parser.add_argument("--n-estimators", type=int,
                        default=DEFAULT_N_ESTIMATORS,
                        help="Número de árboles del Random Forest")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help="Profundidad máxima de los árboles "
                             "(sin valor = sin límite)")
    parser.add_argument("--min-samples-leaf", type=int,
                        default=DEFAULT_MIN_SAMPLES_LEAF,
                        help="Mínimo de muestras por hoja")
    # Opciones
    parser.add_argument("--no-tune", action="store_true",
                        help="Omitir GridSearchCV (más rápido)")
    # MLflow
    parser.add_argument("--no-mlflow", action="store_true",
                        help="No registrar las corridas en MLflow")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT,
                        help="Nombre del experimento en MLflow")
    parser.add_argument("--tracking-uri", default=None,
                        help="URI del servidor MLflow, p. ej. "
                             "http://localhost:5000 (por defecto: ./mlruns)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
