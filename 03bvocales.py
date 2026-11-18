import csv
import numpy as np
from pathlib import Path
from scipy.linalg import solve_toeplitz
from scipy.signal import windows

# ============================================================
# PROGRAMA 06: ESTIMACIÓN DE FORMANTES MEDIANTE LPC
# ============================================================

CARPETA_CAPTURAS = Path("capturas_vocales")
CARPETA_RESULTADOS = Path("resultados_formantes")

CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Parámetros de análisis
INICIO_FRAME = 0.80       # Inicio del fragmento, en segundos
DURACION_FRAME = 0.030    # 30 ms
ORDEN_LPC = 16            # Orden inicial del modelo LPC
F_MIN = 90.0              # Frecuencia mínima candidata (Hz)
F_MAX = 5000.0            # Frecuencia máxima candidata (Hz)
ANCHO_BANDA_MAX = 400.0   # Ancho de banda máximo (Hz)

VOCALES = ["a", "e", "i", "o", "u"]


# ------------------------------------------------------------
# 1. Estimación de coeficientes LPC
# ------------------------------------------------------------

def calcular_lpc(senal, orden):
    """
    Calcula los coeficientes LPC usando las ecuaciones
    de autocorrelación (Yule-Walker).
    """
    senal = np.asarray(senal, dtype=np.float64)

    autocorrelacion = np.correlate(
        senal, senal, mode="full"
    )[len(senal) - 1:]

    if autocorrelacion[0] <= 1e-12:
        raise ValueError("Energía insuficiente en el fragmento.")

    r = autocorrelacion[:orden + 1]

    # Resolver las ecuaciones de Yule-Walker.
    coeficientes_intermedios = solve_toeplitz(
        r[:orden],
        r[1:orden + 1]
    )

    # Polinomio LPC: A(z) = 1 - a1*z^-1 - ...
    coeficientes = np.concatenate((
        [1.0],
        -coeficientes_intermedios
    ))

    return coeficientes


# ------------------------------------------------------------
# 2. Obtener frecuencias y anchos de banda candidatos
# ------------------------------------------------------------

def estimar_formantes(coeficientes, fs):
    """
    Obtiene frecuencias candidatas a formantes a partir
    de las raíces del polinomio LPC.
    """
    raices = np.roots(coeficientes)

    candidatos = []

    for raiz in raices:
        # Considerar una sola raíz de cada par conjugado.
        if np.imag(raiz) <= 0:
            continue

        frecuencia = np.angle(raiz) * fs / (2 * np.pi)

        radio = np.abs(raiz)

        if radio <= 0 or radio >= 1:
            continue

        ancho_banda = (
            -fs / np.pi * np.log(radio)
        )

        if (
            F_MIN <= frecuencia <= min(F_MAX, fs / 2)
            and ancho_banda > 0
            and ancho_banda <= ANCHO_BANDA_MAX
        ):
            candidatos.append(
                (float(frecuencia), float(ancho_banda))
            )

    candidatos.sort(key=lambda x: x[0])

    return candidatos


# ------------------------------------------------------------
# 3. Procesar una grabación
# ------------------------------------------------------------

def procesar_archivo(archivo):
    with np.load(archivo, allow_pickle=False) as datos:
        fs = int(datos["fs"])
        senal = np.asarray(
            datos["normalizada"], dtype=np.float64
        )

        vocal = str(datos["vocal"].item())
        hablante = str(datos["hablante"].item())
        repeticion = int(datos["repeticion"].item())

    if senal.ndim != 1 or not np.all(np.isfinite(senal)):
        raise ValueError("La señal no es válida.")

    inicio = int(INICIO_FRAME * fs)
    longitud = int(DURACION_FRAME * fs)
    fin = inicio + longitud

    if fin > len(senal):
        raise ValueError(
            f"El fragmento solicitado excede la duración "
            f"de {archivo.name}."
        )

    fragmento = senal[inicio:fin].copy()
    fragmento -= np.mean(fragmento)

    if np.max(np.abs(fragmento)) < 1e-8:
        raise ValueError(
            f"Amplitud insuficiente en {archivo.name}."
        )

    # Ventana Hamming, habitual en análisis LPC de voz.
    fragmento *= windows.hamming(len(fragmento), sym=False)

    orden = min(ORDEN_LPC, len(fragmento) // 2 - 1)

    coeficientes = calcular_lpc(fragmento, orden)
    candidatos = estimar_formantes(coeficientes, fs)

    # Los primeros candidatos ordenados por frecuencia
    # se reportan como estimaciones provisionales.
    formantes = candidatos[:3]

    resultado = {
        "archivo": archivo.name,
        "hablante": hablante,
        "vocal": vocal,
        "repeticion": repeticion,
        "fs": fs,
        "inicio_frame_s": INICIO_FRAME,
        "duracion_frame_s": DURACION_FRAME,
        "orden_lpc": orden,
        "F1_Hz": formantes[0][0] if len(formantes) > 0 else "",
        "B1_Hz": formantes[0][1] if len(formantes) > 0 else "",
        "F2_Hz": formantes[1][0] if len(formantes) > 1 else "",
        "B2_Hz": formantes[1][1] if len(formantes) > 1 else "",
        "F3_Hz": formantes[2][0] if len(formantes) > 2 else "",
        "B3_Hz": formantes[2][1] if len(formantes) > 2 else "",
        "numero_candidatos": len(candidatos),
    }

    return resultado


# ------------------------------------------------------------
# 4. Seleccionar grabaciones
# ------------------------------------------------------------

hablante = input(
    "Identificador del hablante (ej. H01): "
).strip().upper()

vocal = input(
    "Vocal que se analizará (a/e/i/o/u): "
).strip().lower()

cantidad_texto = input(
    "Número de repeticiones que se analizarán (ej. 3): "
).strip()

if vocal not in VOCALES:
    raise ValueError("Vocal no válida.")

if not cantidad_texto.isdigit() or int(cantidad_texto) < 1:
    raise ValueError("Introduce una cantidad válida.")

cantidad = int(cantidad_texto)

resultados = []

# ------------------------------------------------------------
# 5. Procesar cada repetición individualmente
# ------------------------------------------------------------

for repeticion in range(1, cantidad + 1):
    patron = (
        f"{hablante}_{vocal}_rep{repeticion:02d}_*.npz"
    )

    archivos = sorted(CARPETA_CAPTURAS.glob(patron))

    if not archivos:
        print(
            f"[AVISO] No se encontró la repetición "
            f"{repeticion}."
        )
        continue

    archivo = archivos[-1]

    try:
        resultado = procesar_archivo(archivo)
        resultados.append(resultado)

        print(
            f"[OK] {archivo.name}: "
            f"F1={resultado['F1_Hz']} Hz, "
            f"F2={resultado['F2_Hz']} Hz, "
            f"F3={resultado['F3_Hz']} Hz"
        )

    except (ValueError, np.linalg.LinAlgError) as error:
        print(f"[ERROR] {archivo.name}: {error}")


# ------------------------------------------------------------
# 6. Guardar resultados individuales en CSV
# ------------------------------------------------------------

if not resultados:
    raise RuntimeError(
        "No se pudo procesar ninguna grabación. "
        "Verifica los archivos y el intervalo de análisis."
    )

archivo_csv = CARPETA_RESULTADOS / (
    f"{hablante}_{vocal}_formantes.csv"
)

campos = list(resultados[0].keys())

with open(
    archivo_csv, "w", newline="", encoding="utf-8-sig"
) as archivo:
    escritor = csv.DictWriter(
        archivo, fieldnames=campos
    )
    escritor.writeheader()
    escritor.writerows(resultados)


# ------------------------------------------------------------
# 7. Calcular estadísticas descriptivas
# ------------------------------------------------------------

archivo_estadisticas = CARPETA_RESULTADOS / (
    f"{hablante}_{vocal}_estadisticas.csv"
)

campos_formantes = ["F1_Hz", "F2_Hz", "F3_Hz"]
estadisticas = []

for campo in campos_formantes:
    valores = []

    for resultado in resultados:
        valor = resultado[campo]

        if valor != "":
            valores.append(float(valor))

    if valores:
        estadisticas.append({
            "hablante": hablante,
            "vocal": vocal,
            "formante": campo,
            "n_muestras": len(valores),
            "promedio_Hz": float(np.mean(valores)),
            "desviacion_estandar_Hz": (
                float(np.std(valores, ddof=1))
                if len(valores) > 1 else 0.0
            ),
        })

if estadisticas:
    with open(
        archivo_estadisticas,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(estadisticas[0].keys())
        )
        escritor.writeheader()
        escritor.writerows(estadisticas)


# ------------------------------------------------------------
# 8. Resumen final
# ------------------------------------------------------------

print("\n====================================")
print("ANÁLISIS LPC TERMINADO")
print("====================================")
print(f"Hablante: {hablante}")
print(f"Vocal: {vocal.upper()}")
print(f"Grabaciones procesadas: {len(resultados)}")
print(f"Resultados individuales: {archivo_csv}")

if estadisticas:
    print(f"Estadísticas: {archivo_estadisticas}")

    for estadistica in estadisticas:
        print(
            f"{estadistica['formante']}: "
            f"promedio = {estadistica['promedio_Hz']:.1f} Hz, "
            f"desviación estándar = "
            f"{estadistica['desviacion_estandar_Hz']:.1f} Hz"
        )

print("\nNota: las frecuencias son estimaciones LPC.")
print("Conviene validar los candidatos con el espectro")
print("y comprobar que correspondan a formantes reales.")

