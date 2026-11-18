import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from datetime import datetime
from pathlib import Path

# ==========================================
# 0. PARAMETROS DE CAPTURA
# ==========================================
FS = 16000
DURACION = 3
CANALES = 1
F_CORTE = 70
ORDEN_FILTRO = 4

VOCALES = ["a", "e", "i", "o", "u"]
CARPETA = Path("capturas_vocales")
CARPETA.mkdir(exist_ok=True)

print("RECONOCIMIENTO DE VOCALES - ETAPA 1")
print("-----------------------------------")
print(f"Frecuencia de muestreo: {FS} Hz")
print(f"Duracion de captura: {DURACION} s")
print(f"Canales: {CANALES}")
print()

# ==========================================
# 1. SELECCION DE LA VOCAL Y LA MUESTRA
# ==========================================
vocal = input(
    "¿Que vocal vas a pronunciar? (a, e, i, o, u): "
).strip().lower()

if vocal not in VOCALES:
    print("Vocal no valida. Programa terminado.")
    raise SystemExit

hablante = input(
    "Identificador del hablante (ej. H01): "
).strip().upper()

if not hablante:
    hablante = "H01"

repeticion_texto = input(
    "Numero de repeticion (1, 2, 3...): "
).strip()

try:
    repeticion = int(repeticion_texto)
    if repeticion < 1:
        raise ValueError
except ValueError:
    print("La repeticion debe ser un entero positivo.")
    raise SystemExit

# ==========================================
# 2. CAPTURA DE VOZ
# ==========================================
input(
    "\nCuando estes listo, presiona ENTER y "
    "pronuncia la vocal sostenida..."
)

print("\nGrabando...")

try:
    audio = sd.rec(
        int(FS * DURACION),
        samplerate=FS,
        channels=CANALES,
        dtype="float32"
    )
    sd.wait()
except Exception as error:
    print(f"Error durante la captura: {error}")
    raise SystemExit

x = audio[:, 0].astype(np.float64)

if len(x) < FS:
    print("Captura demasiado corta. No se guardo.")
    raise SystemExit

print("Grabacion terminada.")
print(f"Muestras capturadas: {len(x)}")

# ==========================================
# 3. VALIDACION DE LA SEÑAL ORIGINAL
# ==========================================
if not np.all(np.isfinite(x)):
    print("La señal contiene valores invalidos. No se guardo.")
    raise SystemExit

pico_original = np.max(np.abs(x))
rms_original = np.sqrt(np.mean(x ** 2))

# Umbrales orientativos para audio float normalizado.
# Pueden variar segun el microfono y su ganancia.
UMBRAL_SENAL_DEBIL = 1e-4
UMBRAL_SATURACION = 0.99

if rms_original < UMBRAL_SENAL_DEBIL:
    print(
        "ADVERTENCIA: señal muy debil o casi silenciosa. "
        "Revisa el microfono y su nivel de entrada."
    )

if pico_original >= UMBRAL_SATURACION:
    print(
        "ADVERTENCIA: posible saturacion. "
        "Reduce la ganancia o aleja el microfono."
    )

print(f"Pico original: {pico_original:.5f}")
print(f"RMS original: {rms_original:.5f}")

# ==========================================
# 4. ELIMINAR COMPONENTE CONTINUA
# ==========================================
x_dc = x - np.mean(x)

# ==========================================
# 5. FILTRO PASAALTAS
# ==========================================
sos = butter(
    ORDEN_FILTRO,
    F_CORTE,
    btype="highpass",
    fs=FS,
    output="sos"
)

try:
    x_filtrada = sosfiltfilt(sos, x_dc)
except ValueError as error:
    print(f"No se pudo filtrar la señal: {error}")
    raise SystemExit

# ==========================================
# 6. NORMALIZACION DE AMPLITUD
# ==========================================
pico_filtrado = np.max(np.abs(x_filtrada))

if pico_filtrado > 1e-8:
    x_normalizada = x_filtrada / pico_filtrado
else:
    print("Señal insuficiente despues del filtrado. No se guardo.")
    raise SystemExit

rms_filtrada = np.sqrt(np.mean(x_filtrada ** 2))
rms_normalizada = np.sqrt(np.mean(x_normalizada ** 2))

# ==========================================
# 7. MEDIDAS Y RESUMEN
# ==========================================
duracion_real = len(x) / FS
tiempo = np.arange(len(x)) / FS

print("\nRESUMEN DE LA CAPTURA")
print("---------------------")
print(f"Duracion real: {duracion_real:.3f} s")
print(f"RMS original: {rms_original:.5f}")
print(f"RMS filtrada: {rms_filtrada:.5f}")
print(f"RMS normalizada: {rms_normalizada:.5f}")

# ==========================================
# 8. GRAFICAR LAS SEÑALES
# ==========================================
fig, ejes = plt.subplots(3, 1, figsize=(12, 8))

ejes[0].plot(tiempo, x, linewidth=0.7)
ejes[0].set_title("1. Señal original")
ejes[0].set_ylabel("Amplitud")
ejes[0].grid(True, alpha=0.3)

ejes[1].plot(tiempo, x_filtrada, linewidth=0.7)
ejes[1].set_title("2. Señal filtrada (pasaaltas de 70 Hz)")
ejes[1].set_ylabel("Amplitud")
ejes[1].grid(True, alpha=0.3)

ejes[2].plot(tiempo, x_normalizada, linewidth=0.7)
ejes[2].set_title("3. Señal filtrada y normalizada")
ejes[2].set_xlabel("Tiempo (s)")
ejes[2].set_ylabel("Amplitud")
ejes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 9. CONFIRMAR Y GUARDAR LA CAPTURA
# ==========================================
confirmacion = input(
    "\n¿La grabacion se ve y se escucha razonablemente bien? "
    "(s/n): "
).strip().lower()

if confirmacion != "s":
    print("Captura descartada. Vuelve a grabar la vocal.")
    raise SystemExit

marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")

nombre_archivo = (
    f"{hablante}_{vocal}_rep{repeticion:02d}_{marca_tiempo}.npz"
)

ruta = CARPETA / nombre_archivo

np.savez(
    ruta,
    fs=FS,
    vocal=vocal,
    hablante=hablante,
    repeticion=repeticion,
    fecha_hora=marca_tiempo,
    original=x,
    filtrada=x_filtrada,
    normalizada=x_normalizada,
    pico_original=pico_original,
    rms_original=rms_original,
    rms_filtrada=rms_filtrada
)

print("\nCaptura guardada correctamente.")
print(f"Archivo: {ruta}")
print("La señal original, filtrada y normalizada quedaron almacenadas.")

