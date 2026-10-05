# TC5061 - Experimentación con MLFlow

## Descripción

Este proyecto implementa un flujo de experimentación de Machine Learning utilizando MLFlow Tracking sobre el dataset Breast Cancer Wisconsin.

Se utiliza un modelo de Regresión Logística para realizar la clasificación y comparar distintas configuraciones de hiperparámetros.

## Modelo

Regresión Logística utilizando scikit-learn.

## Hiperparámetros

El script permite modificar desde la línea de comandos:

- C
- max_iter
- solver
- random_state

## Ejecución

Ejemplo:

python train.py --C 1.0 --max_iter 1000 --solver liblinear --random_state 42

## MLFlow

Cada ejecución registra:

- Hiperparámetros
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matriz de confusión
- Archivo de métricas
- Modelo entrenado
# TC5061 - Experimentación con MLFlow

## Descripción

Este proyecto implementa un flujo de experimentación de Machine Learning utilizando MLFlow Tracking sobre el dataset Breast Cancer Wisconsin.

Se utiliza un modelo de Regresión Logística para realizar la clasificación y comparar distintas configuraciones de hiperparámetros.

## Modelo

Regresión Logística utilizando scikit-learn.

## Hiperparámetros

El script permite modificar desde la línea de comandos:

- C
- max_iter
- solver
- random_state

## Ejecución

Ejemplo:

python train.py --C 1.0 --max_iter 1000 --solver liblinear --random_state 42

## MLFlow

Cada ejecución registra:

- Hiperparámetros
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matriz de confusión
- Archivo de métricas
- Modelo entrenado

Las ejecuciones se agrupan en el experimento:

Regresión logística del cáncer de mama

## Dependencias

Instalar las dependencias con:

pip install -r requirements.txt

## Reproducibilidad

La semilla aleatoria se controla mediante el parámetro `random_state`, permitiendo reproducir los resultados utilizando los mismos parámetros.
