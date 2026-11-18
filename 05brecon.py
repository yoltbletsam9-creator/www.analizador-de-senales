import joblib
import numpy as np
import sounddevice as sd



from scipy.signal import butter
from scipy.signal import sosfiltfilt
from scipy.signal import windows
from scipy.linalg import solve_toeplitz



# ==========================================
# CONFIGURACION
# ==========================================



FS = 16000
DURACION = 3
CANALES = 1



F_CORTE = 70
ORDEN_FILTRO = 4



INICIO_FRAME = 0.80
DURACION_FRAME = 0.03



ORDEN_LPC = 16



F_MIN = 90
F_MAX = 5000
ANCHO_BANDA_MAX = 400



# ==========================================
# CARGAR MODELO
# ==========================================



datos_modelo = joblib.load(
    "modelo_vocales_knn.pkl"
)



modelo = datos_modelo["modelo"]
scaler = datos_modelo["scaler"]



# ==========================================
# FUNCIONES LPC
# ==========================================



def calcular_lpc(signal, orden):



    autocorr = np.correlate(
        signal,
        signal,
        mode="full"
    )[len(signal)-1:]



    r = autocorr[:orden+1]



    a = solve_toeplitz(
        r[:orden],
        r[1:orden+1]
    )



    return np.concatenate(
        ([1.0], -a)
    )




def obtener_formantes(
    coeficientes,
    fs
):



    raices = np.roots(
        coeficientes
    )



    candidatos = []



    for raiz in raices:



        if np.imag(raiz) <= 0:
            continue



        frecuencia = (
            np.angle(raiz)
            * fs
            / (2*np.pi)
        )



        radio = np.abs(raiz)



        if radio <= 0 or radio >= 1:
            continue



        banda = (
            -fs
            / np.pi
            * np.log(radio)
        )



        if (
            F_MIN <= frecuencia <= F_MAX
            and banda <= ANCHO_BANDA_MAX
        ):
            candidatos.append(
                frecuencia
            )



    candidatos.sort()



    return candidatos[:3]



# ==========================================
# CAPTURA
# ==========================================



input(
    "Presiona ENTER y pronuncia una vocal..."
)



print()
print("Grabando...")



audio = sd.rec(
    int(FS * DURACION),
    samplerate=FS,
    channels=CANALES,
    dtype="float32"
)



sd.wait()



print("Grabacion terminada.")



x = audio[:, 0].astype(
    np.float64
)



# ==========================================
# PREPROCESAMIENTO
# ==========================================



x -= np.mean(x)



sos = butter(
    ORDEN_FILTRO,
    F_CORTE,
    btype="highpass",
    fs=FS,
    output="sos"
)



x = sosfiltfilt(
    sos,
    x
)



x /= np.max(np.abs(x))



# ==========================================
# FRAME LPC
# ==========================================



inicio = int(
    INICIO_FRAME * FS
)



fin = inicio + int(
    DURACION_FRAME * FS
)



fragmento = x[inicio:fin]



fragmento -= np.mean(
    fragmento
)



fragmento *= windows.hamming(
    len(fragmento),
    sym=False
)



coeficientes = calcular_lpc(
    fragmento,
    ORDEN_LPC
)



formantes = obtener_formantes(
    coeficientes,
    FS
)



if len(formantes) < 3:
    raise ValueError(
        "No fue posible calcular F1 F2 F3."
    )



F1 = formantes[0]
F2 = formantes[1]
F3 = formantes[2]



print()
print("FORMANTES")
print("----------")
print(f"F1 = {F1:.1f} Hz")
print(f"F2 = {F2:.1f} Hz")
print(f"F3 = {F3:.1f} Hz")



# ==========================================
# CLASIFICACION
# ==========================================



muestra = scaler.transform([
    [F1, F2, F3]
])



vocal = modelo.predict(
    muestra
)[0]



print()
print("===================")
print("VOCAL DETECTADA")
print("===================")
print(vocal.upper())
print("===================")
print("===================")