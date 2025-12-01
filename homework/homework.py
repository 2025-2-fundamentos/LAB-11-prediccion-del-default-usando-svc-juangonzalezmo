# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline as build_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA


def cargar_datos(nombre_archivo):
    ruta_base = Path("files/input")
    return pd.read_csv(ruta_base / nombre_archivo, compression="zip")


def depurar_datos(data: pd.DataFrame):
    df = data.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns="ID")
    df = df.dropna()
    df = df[df["MARRIAGE"] != 0]
    df = df[df["EDUCATION"] != 0]
    df["EDUCATION"] = df["EDUCATION"].map(lambda valor: 4 if valor > 4 else valor)
    return df


def separar_variables(data: pd.DataFrame):
    caracteristicas = data.drop(columns="default").copy()
    objetivo = data["default"].copy()
    return caracteristicas, objetivo


def dividir_datos(data: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    caracteristicas, objetivo = separar_variables(data)
    return train_test_split(caracteristicas, objetivo, random_state=0)


def pipeline():
    transformador = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["SEX", "EDUCATION", "MARRIAGE"],
            )
        ],
        remainder=StandardScaler(),
    )

    modelo_svm = build_pipeline(
        transformador,
        PCA(),
        SelectKBest(k=12),
        SVC(gamma=0.1),
    )

    return modelo_svm


def grid_search(modelo, parametros, cv=10):
    return GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )


def guardar_modelo(modelo):
    import pickle
    import gzip

    ruta_modelos = Path("files/models")
    ruta_modelos.mkdir(exist_ok=True)

    with gzip.open(ruta_modelos / "model.pkl.gz", "wb") as archivo:
        pickle.dump(modelo, archivo)


def guardar_metricas(lista_metricas):
    import json

    ruta_salida = Path("files/output")
    ruta_salida.mkdir(exist_ok=True)

    with open(ruta_salida / "metrics.json", "w") as archivo:
        lineas = [json.dumps(metrica) + "\n" for metrica in lista_metricas]
        archivo.writelines(lineas)


def entrenar_modelo(caracteristicas_entrenamiento, objetivo_entrenamiento):
    modelo_base = pipeline()

    busqueda = grid_search(
        modelo_base,
        {
            "pca__n_components": [20, 21],
        },
    )

    busqueda.fit(caracteristicas_entrenamiento, objetivo_entrenamiento)
    guardar_modelo(busqueda)
    return busqueda


def metricas(valores_reales, valores_estimados):
    from sklearn.metrics import (
        precision_score,
        balanced_accuracy_score,
        recall_score,
        f1_score,
    )

    precision = precision_score(valores_reales, valores_estimados)
    balanced_acc = balanced_accuracy_score(valores_reales, valores_estimados)
    sensibilidad = recall_score(valores_reales, valores_estimados)
    puntaje_f1 = f1_score(valores_reales, valores_estimados)

    return precision, balanced_acc, sensibilidad, puntaje_f1


def matriz_c(nombre_conjunto, valores_reales, valores_estimados):
    from sklearn.metrics import confusion_matrix

    matriz = confusion_matrix(valores_reales, valores_estimados)

    return {
        "type": "cm_matrix",
        "dataset": nombre_conjunto,
        "true_0": {
            "predicted_0": int(matriz[0][0]),
            "predicted_1": int(matriz[0][1]),
        },
        "true_1": {
            "predicted_0": int(matriz[1][0]),
            "predicted_1": int(matriz[1][1]),
        },
    }


def ejecutar_proceso():
    df_entrenamiento = depurar_datos(cargar_datos("train_data.csv.zip"))
    df_prueba = depurar_datos(cargar_datos("test_data.csv.zip"))

    x_entrenamiento, y_entrenamiento = separar_variables(df_entrenamiento)
    x_prueba, y_prueba = separar_variables(df_prueba)

    modelo_entrenado = entrenar_modelo(x_entrenamiento, y_entrenamiento)

    resultados_metricas = []
    for nombre_conjunto, x, y in [
        ("train", x_entrenamiento, y_entrenamiento),
        ("test", x_prueba, y_prueba),
    ]:
        predicciones = modelo_entrenado.predict(x)
        precision, balanced_acc, sensibilidad, puntaje_f1 = metricas(y, predicciones)
        resultados_metricas.append(
            {
                "type": "metrics",
                "dataset": nombre_conjunto,
                "precision": precision,
                "balanced_accuracy": balanced_acc,
                "recall": sensibilidad,
                "f1_score": puntaje_f1,
            }
        )

    matrices_confusion = [
        matriz_c(nombre_conjunto, y, modelo_entrenado.predict(x))
        for nombre_conjunto, x, y in [
            ("train", x_entrenamiento, y_entrenamiento),
            ("test", x_prueba, y_prueba),
        ]
    ]

    guardar_metricas(resultados_metricas + matrices_confusion)


ejecutar_proceso()
