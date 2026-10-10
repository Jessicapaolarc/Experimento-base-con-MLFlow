# Predicción de la calidad del vino: entrenamiento con MLflow

Script `train.py` migrado del notebook `TC5061_Notebook_Baseline_RuizJessica.ipynb`.
El notebook se conserva solo para la exploración de datos (EDA).

## Estructura del script

| Función | Responsabilidad |
|---|---|
| `set_global_seed` | Fija las semillas de Python y NumPy |
| `load_data` | Lee el CSV (`Id` como índice) |
| `clean_data` | Elimina duplicados; conserva outliers |
| `split_features_target`, `split_data` | Separa X e y; división 80/20 estratificada |
| `train_dummy`, `train_linear_regression`, `train_random_forest`, `tune_random_forest` | Entrenamiento de los modelos |
| `compute_metrics`, `evaluate_models` | RMSE, MAE y R² (prueba, entrenamiento y validación cruzada) |
| `log_run`, `log_to_mlflow` | Registro en MLflow: parámetros, métricas, artefactos y modelo |

## Cómo ejecutarlo

```bash
python train.py                                   # valores por defecto
python train.py --n-estimators 500 --max-depth 8 --min-samples-leaf 3 --random-state 7
python train.py --tracking-uri http://localhost:5000 --experiment-name mi_experimento
python train.py --no-tune --no-mlflow             # rápido y sin registro
```

| Argumento | Descripción | Por defecto |
|---|---|---|
| `--n-estimators` | Árboles del Random Forest | 300 |
| `--max-depth` | Profundidad máxima | sin límite |
| `--min-samples-leaf` | Mínimo de muestras por hoja | 1 |
| `--random-state` | Semilla aleatoria | 42 |
| `--test-size` | Proporción del conjunto de prueba | 0.2 |
| `--data-path` | Ruta del CSV | `wine_quality_df.csv` |
| `--no-tune` | Omite `GridSearchCV` | desactivado |
| `--no-mlflow` | No registra en MLflow | desactivado |
| `--experiment-name` | Nombre del experimento | `wine_quality` |
| `--tracking-uri` | Servidor MLflow | carpeta local `mlruns/` |

## Fijación de la semilla

La semilla (`--random-state`, 42 por defecto) se aplica en todos los puntos con
aleatoriedad:

| Dónde | Qué controla |
|---|---|
| `set_global_seed` | Semillas globales de `random` y `numpy` |
| `train_test_split(random_state=...)` | Qué filas van a entrenamiento y prueba |
| `RandomForestRegressor(random_state=...)` | Muestreo de datos y variables en cada árbol (modelo sin ajustar y modelo ajustado) |
| `KFold(random_state=...)` | Particiones de la validación cruzada y de `GridSearchCV` |
| Gráfica `real_vs_predicho.png` | Ruido visual (jitter) con semilla fija |

La semilla usada también queda registrada como parámetro en cada corrida de MLflow
(`random_state`).

### Comprobar que es reproducible

Dos ejecuciones con los mismos argumentos deben dar exactamente la misma salida:

```bash
python train.py --no-tune --no-mlflow > corrida_a.txt
python train.py --no-tune --no-mlflow > corrida_b.txt
diff corrida_a.txt corrida_b.txt && echo "Resultados idénticos"
```

Si el comando imprime `Resultados idénticos`, la ejecución es reproducible. Con otra
semilla (`--random-state 7`) los resultados cambian, como es esperable.

Condiciones de la reproducibilidad: mismos datos, mismos argumentos y **mismas versiones
de las librerías** (ver siguiente sección). Con versiones distintas de scikit-learn o
NumPy los números pueden variar ligeramente.

## Dependencias

Las versiones exactas utilizadas están en `requirements.txt`.

### Instalación en un entorno limpio

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Cómo se generó `requirements.txt`

Desde el entorno virtual donde se ejecutó el proyecto:

```bash
{ echo "# Python $(python --version | cut -d' ' -f2)"; \
  pip freeze | grep -iE "^(pandas|numpy|scikit-learn|matplotlib|mlflow)=="; } > requirements.txt
```

Solo se fijan las librerías que usa el código directamente; pip resuelve el resto
de dependencias. Para un registro completo de todas las versiones instaladas:

```bash
pip freeze > requirements-lock.txt
```
