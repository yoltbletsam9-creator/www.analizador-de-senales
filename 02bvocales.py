import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import windows

# ============================================================
# PROGRAMA 03: GENERACIÓN DEL ESPECTRO DE UNA VOCAL
# ============================================================

CARPETA_CAPTURAS = Path("capturas_vocales")
CARPETA_ESPECTROS = Path("espectros_vocales")

CARPETA_ESPECTROS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Seleccionar la grabación
# ------------------------------------------------------------

hablante = input("Identificador del hablante (ej. H01): ").strip().upper()
vocal = input("Vocal que se analizará (a/e/i/o/u): ").strip().lower()

if vocal not in ["a", "e", "i", "o", "u"]:
    raise ValueError("La vocal debe ser a, e, i, o u.")

repeticion = input("Número de repetición (1 a 10): ").strip()

if not repeticion.isdigit() or int(repeticion) < 1:
    raise ValueError("Introduce un número de repetición válido.")

repeticion = int(repeticion)

patron = (
    f"{hablante}_{vocal}_rep{repeticion:02d}_*.npz"
)
archivos = sorted(CARPETA_CAPTURAS.glob(patron))

if not archivos:
    raise FileNotFoundError(
        f"No se encontró una grabación con el patrón:\n{patron}\n"
        f"Verifica que exista en la carpeta "
        f"'{CARPETA_CAPTURAS}'."
    )

# Si hubiera más de una grabación con el mismo número,
# utilizamos la más reciente según el nombre del archivo.
archivo_entrada = archivos[-1]

print(f"\nGrabación seleccionada: {archivo_entrada.name}")

# ------------------------------------------------------------
# 2. Cargar los datos de la grabación
# ------------------------------------------------------------

with np.load(archivo_entrada, allow_pickle=False) as datos:
    fs = int(datos["fs"])
    senal = np.asarray(datos["normalizada"], dtype=np.float64)
    vocal_guardada = str(datos["vocal"].item())
    hablante_guardado = str(datos["hablante"].item())
    repeticion_guardada = int(datos["repeticion"].item())

if senal.ndim != 1 or senal.size == 0:
    raise ValueError("La señal debe ser un vector no vacío.")

if not np.all(np.isfinite(senal)):
    raise ValueError("La señal contiene valores no válidos.")

# ------------------------------------------------------------
# 3. Seleccionar un fragmento estable de la vocal
# ------------------------------------------------------------

# Intervalo inicial recomendado para inspección:
# desde 0.50 s hasta 1.20 s.
# Ajustar si la vocal sostenida comienza o termina en
# otro momento de la grabación.

INICIO = 0.50
FIN = 1.20

i_inicio = int(INICIO * fs)
i_fin = int(FIN * fs)

if i_fin > len(senal) or i_inicio >= i_fin:
    raise ValueError(
        "El intervalo seleccionado excede la duración "
        "de la grabación."
    )

fragmento = senal[i_inicio:i_fin].copy()

# Eliminar cualquier componente continua residual.
fragmento -= np.mean(fragmento)

if np.max(np.abs(fragmento)) < 1e-8:
    raise ValueError(
        "El fragmento tiene amplitud insuficiente. "
        "Selecciona otro intervalo."
    )

# ------------------------------------------------------------
# 4. Aplicar ventana Hann y calcular la FFT
# ------------------------------------------------------------

ventana = windows.hann(len(fragmento), sym=False)
senal_ventaneada = fragmento * ventana

# La FFT se calcula con una longitud igual o superior
# a la longitud del fragmento, usando una potencia de dos.
n_fft = 2 ** int(np.ceil(np.log2(len(fragmento))))

espectro_complejo = np.fft.rfft(
    senal_ventaneada,
    n=n_fft
)

frecuencias = np.fft.rfftfreq(
    n_fft,
    d=1.0 / fs
)

# Magnitud del espectro.
magnitud = np.abs(espectro_complejo)

# Magnitud relativa en decibelios.
# El máximo del espectro se establece en 0 dB.
magnitud_db = 20 * np.log10(
    np.maximum(magnitud / (np.max(magnitud) + 1e-12), 1e-8)
)

# ------------------------------------------------------------
# 5. Guardar el espectro
# ------------------------------------------------------------

nombre_salida = (
    f"{hablante_guardado}_{vocal_guardada}_"
    f"rep{repeticion_guardada:02d}_espectro.npz"
)

archivo_salida = CARPETA_ESPECTROS / nombre_salida

np.savez(
    archivo_salida,
    fs=fs,
    vocal=vocal_guardada,
    hablante=hablante_guardado,
    repeticion=repeticion_guardada,
    archivo_origen=archivo_entrada.name,
    inicio_analisis=INICIO,
    fin_analisis=FIN,
    frecuencias=frecuencias,
    magnitud=magnitud,
    magnitud_db=magnitud_db
)

print("\nAnálisis terminado.")
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Duración analizada: {len(fragmento) / fs:.3f} s")
print(f"Puntos de la FFT: {n_fft}")
print(f"Resolución de la FFT: {fs / n_fft:.2f} Hz")
print(f"Espectro guardado en: {archivo_salida}")

# ------------------------------------------------------------
# 6. Graficar el espectro
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(frecuencias, magnitud_db, linewidth=0.8)

plt.xlim(0, min(4000, fs / 2))
plt.ylim(-80, 5)

plt.title(
    f"Espectro de la vocal {vocal_guardada.upper()} "
    f"- {hablante_guardado}, repetición {repeticion_guardada}"
)

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud relativa (dB)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
