import argparse
import json
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def cargar_datos(seed):
    datos = load_breast_cancer()
    X = datos.data
    y = datos.target

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )


def crear_modelo(C, max_iter, solver, seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=seed
        ))
    ])


def evaluar_modelo(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, predicciones),
        "precision": precision_score(y_test, predicciones),
        "recall": recall_score(y_test, predicciones),
        "f1_score": f1_score(y_test, predicciones),
        "roc_auc": roc_auc_score(y_test, probabilidades),
    }

    return metricas, predicciones


def guardar_artefactos(y_test, predicciones, metricas):
    cm = confusion_matrix(y_test, predicciones)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title("Matriz de Confusión")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()

    with open("metrics.json", "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Experimentación de Regresión Logística con MLflow"
    )

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    mlflow.set_experiment("Breast_Cancer_LogisticRegression")

    X_train, X_test, y_train, y_test = cargar_datos(args.seed)

    modelo = crear_modelo(
        args.C,
        args.max_iter,
        args.solver,
        args.seed
    )

    with mlflow.start_run():

        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("seed", args.seed)

        modelo.fit(X_train, y_train)

        metricas, predicciones = evaluar_modelo(
            modelo,
            X_test,
            y_test
        )

        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        guardar_artefactos(
            y_test,
            predicciones,
            metricas
        )

        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("metrics.json")

        mlflow.sklearn.log_model(
            modelo,
            artifact_path="model"
        )

        print("Corrida registrada correctamente.")
        print("Parámetros:", vars(args))
        print("Métricas:", metricas)


if __name__ == "__main__":
    main()
