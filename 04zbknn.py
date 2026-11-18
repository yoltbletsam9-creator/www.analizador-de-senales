import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

CARPETA = Path("resultados_formantes")

X = []
y = []

archivos = list(
    CARPETA.glob("*_formantes.csv")
)

if not archivos:
    raise FileNotFoundError(
        "No se encontraron archivos CSV."
    )

for archivo in archivos:

    vocal = archivo.stem.split("_")[1]

    datos = pd.read_csv(archivo)

    for _, fila in datos.iterrows():

        if (
            pd.notna(fila["F1_Hz"])
            and pd.notna(fila["F2_Hz"])
            and pd.notna(fila["F3_Hz"])
        ):

            X.append([
                fila["F1_Hz"],
                fila["F2_Hz"],
                fila["F3_Hz"]
            ])

            y.append(vocal)

X = np.array(X)

scaler = StandardScaler()

X = scaler.fit_transform(X)

modelo = KNeighborsClassifier(
    n_neighbors=3
)

modelo.fit(X, y)

joblib.dump(
    {
        "modelo": modelo,
        "scaler": scaler
    },
    "modelo_vocales_knn.pkl"
)

print()
print("MODELO ENTRENADO")
print("----------------")
print(f"Muestras: {len(X)}")
print("Archivo generado:")
print("modelo_vocales_knn.pkl")