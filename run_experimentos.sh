#!/usr/bin/env bash
# Ejecuta varias configuraciones de hiperparámetros bajo UN mismo experimento.
# Diseño: una configuración BASE y, a partir de ella, se cambia un solo
# hiperparámetro a la vez. Así se puede atribuir el cambio en las métricas a
# ese hiperparámetro. Una corrida extra cambia solo la semilla para estimar
# cuánto varía el resultado por azar.
#
# Uso (con el entorno virtual activo y mlflow ui corriendo):
#   bash run_experimentos.sh
URI="http://localhost:5000"
EXP="wine_quality_rf_comparacion_hiperparametros"

# n_estimators  max_depth  min_samples_leaf  random_state
CONFIGS=(
  "300  8   3   42"   # BASE
  "50   8   3   42"   # solo cambia n_estimators
  "500  8   3   42"   # solo cambia n_estimators
  "300  3   3   42"   # solo cambia max_depth
  "300  12  3   42"   # solo cambia max_depth
  "300  8   1   42"   # solo cambia min_samples_leaf
  "300  8   10  42"   # solo cambia min_samples_leaf
  "300  8   3   7"    # solo cambia la semilla (variabilidad por azar)
)

for cfg in "${CONFIGS[@]}"; do
  read -r N D L S _ <<< "${cfg%%#*}"
  echo "=== n_estimators=$N max_depth=$D min_samples_leaf=$L random_state=$S ==="
  python train.py --no-tune \
    --tracking-uri "$URI" --experiment-name "$EXP" \
    --n-estimators "$N" --max-depth "$D" \
    --min-samples-leaf "$L" --random-state "$S"
done
echo "Listo: revisa el experimento '$EXP' en $URI"
