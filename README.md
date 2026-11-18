# Detector de Vocales mediante Análisis LPC y Clasificación KNN

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-Academic-blue?style=for-the-badge)

Implementación de un sistema de reconocimiento de vocales mediante captura de audio, procesamiento digital de señales, estimación de formantes LPC y clasificación con el algoritmo K-Nearest Neighbors (KNN).

El proyecto permite grabar las vocales `a`, `e`, `i`, `o` y `u`, visualizar sus espectros de frecuencia, obtener sus formantes F1, F2 y F3, entrenar un modelo de clasificación y reconocer una vocal pronunciada por el microfono.

---

## Descripción

El reconocimiento de vocales es una aplicación del procesamiento de señales de voz y del aprendizaje automático. Este proyecto implementa un flujo completo basado en las siguientes características:

- **Captura de audio:** grabación de 3 segundos a una frecuencia de muestreo de 16 kHz.
- **Procesamiento modular:** el sistema separa la captura, el análisis espectral, la estimación LPC, el entrenamiento y el reconocimiento.
- **Preprocesamiento de la señal:** eliminación de componente continua, filtrado pasaaltas y normalización de amplitud.
- **Análisis espectral:** cálculo de la transformada rápida de Fourier para observar la distribución de frecuencias.
- **Estimación LPC:** obtención de los formantes F1, F2 y F3 asociados a la pronunciación de cada vocal.
- **Clasificación KNN:** entrenamiento de un modelo con las características obtenidas mediante LPC.
- **Persistencia del modelo:** almacenamiento del clasificador y el escalador en formato `.pkl` para reutilizarlos sin entrenar nuevamente.
- **Resultados documentados:** generación de archivos CSV con formantes individuales y estadísticas descriptivas.

## Docente

**Dr. Manuel**

## Integrantes del equipo

- Angel Carrera
- Oliver Urbina
- Herrera Victor
- Isaac Vazquez
- Uriel Vigueras
- Amaro Daniel

## Créditos

El desarrollo de este proyecto fue realizado por los integrantes del equipo mencionados anteriormente.

El Dr. Manuel proporcionó una parte del código base del sistema de reconocimiento de vocales y participó como docente responsable del proyecto. Esta parte inicial sirvió como fundamento para el análisis, la integración y el desarrollo completo del sistema.

Proyecto en Python para captar grabaciones de las vocales `a`, `e`, `i`, `o` y `u`, analizarlas mediante LPC y clasificarlas con KNN.

## Funcionalidad

El flujo consta de cinco programas:

1. `01bcapturavocales.py`: graba una vocal durante 3 segundos, elimina la componente continua, aplica un filtro pasaaltas y guarda la señal original y procesada.
2. `02bvocales.py`: calcula y muestra el espectro FFT de una grabación seleccionada.
3. `03bvocales.py`: procesa las repeticiones de una vocal mediante LPC y obtiene F1, F2 y F3. Genera archivos CSV individuales y estadísticos.
4. `04zbknn.py`: usa los resultados LPC para entrenar un clasificador KNN y genera `modelo_vocales_knn.pkl`.
5. `05brecon.py`: captura una vocal nueva y predice cuál de las cinco vocales es.

Las frecuencias de muestreo se trabajan a 16 kHz y el modelo incluido permite ejecutar el reconocimiento sin volver a entrenar.

## 1. Metodología

El desarrollo del sistema siguió los siguientes pasos:

1. **Captura de voz:** se configuró el microfono a una frecuencia de muestreo de 16 kHz y se grabaron segmentos de 3 segundos.
2. **Identificación de datos:** se solicitó la vocal pronunciada, el identificador del hablante y el número de repetición.
3. **Preprocesamiento:** se eliminó la componente continua, se aplicó un filtro pasaaltas de 70 Hz y se normalizó la amplitud de la señal.
4. **Selección del fragmento:** se tomó un intervalo estable de la grabación para evitar segmentos con silencio o transiciones entre pronunciaciones.
5. **Análisis espectral:** se aplicó una ventana Hann y la transformada rápida de Fourier para obtener el espectro de magnitudes.
6. **Estimación LPC:** se calculó un modelo de predicción lineal de orden 16 y se extrajeron las frecuencias y anchos de banda candidatos a formantes.
7. **Obtención de formantes:** se conservaron los tres primeros formantes representativos, F1, F2 y F3.
8. **Construcción del conjunto de datos:** se almacenaron los formantes en archivos CSV junto con la vocal correspondiente.
9. **Escalamiento:** se estandarizó el conjunto de características antes del entrenamiento para dar una escala comparable a F1, F2 y F3.
10. **Entrenamiento:** se creó un clasificador K-Nearest Neighbors con tres vecinos.
11. **Persistencia del modelo:** se guardaron el modelo y el escalador en `modelo_vocales_knn.pkl`.
12. **Reconocimiento:** se grabó una nueva vocal, se repitió el preprocesamiento y el análisis LPC, y el modelo predijo la vocal correspondiente.

## 2. Glosario de términos

- **Señal de voz:** variación de presión sonora registrada por el microfono y almacenada como muestras digitales.
- **Frecuencia de muestreo:** cantidad de muestras tomadas por segundo. En este proyecto es de 16 000 Hz.
- **Componente continua:** nivel constante o desplazamiento promedio que puede estar presente en la señal y no aporta información útil a la voz.
- **Filtro pasaaltas:** filtro que reduce las frecuencias inferiores a un límite establecido. Aquí elimina componentes por debajo de 70 Hz.
- **Normalización:** ajuste de amplitud de la señal para que su valor máximo sea comparable entre grabaciones.
- **FFT:** Transformada Rápida de Fourier; permite pasar una señal del dominio del tiempo al dominio de la frecuencia.
- **Espectro:** representación de las frecuencias presentes en una señal y de su amplitud.
- **Ventana Hann:** función aplicada a un fragmento antes de calcular la FFT para reducir pérdidas o discontinuidades artificiales en sus extremos.
- **LPC:** Linear Predictive Coding o codificación predictiva lineal. Estima cómo puede modelarse el tracto vocal mediante coeficientes.
- **Formante:** concentración de energía que aparece en ciertas regiones de frecuencia debido a la configuración del tracto vocal.
- **F1:** primer formante, normalmente asociado a características vocales de baja frecuencia.
- **F2:** segundo formante, útil para distinguir la posición de la lengua en la pronunciación.
- **F3:** tercer formante, usado junto con F1 y F2 como característica adicional de clasificación.
- **Raíz del polinomio LPC:** valor complejo usado para calcular las frecuencias y los anchos de banda asociados a posibles formantes.
- **CSV:** archivo de texto separado por comas utilizado para almacenar los formantes y las estadísticas.
- **Escalador:** transformación que normaliza las características antes de entrenar y consultar el modelo.
- **KNN:** K-Nearest Neighbors, o K vecinos más cercanos. Clasifica una muestra según las clases de las muestras conocidas más próximas.
- **Modelo:** estructura entrenada que contiene las reglas y datos necesarios para realizar una clasificación.
- **Pickle/Joblib:** formato usado con Joblib para guardar y cargar el modelo y el escalador.
- **NPZ:** archivo binario de NumPy empleado para guardar señales, frecuencias y metadatos de las grabaciones.

## 3. Diseño de la aplicación

La aplicación se divide en etapas independientes que permiten verificar el proceso antes de realizar la clasificación:

```text
Microfono
   |
   v
Captura de 3 segundos a 16 kHz
   |
   v
Eliminacion de continua + filtro pasaaltas + normalizacion
   |
   v
Seleccion de fragmento estable
   |
   +--------------------------+
   |                          |
   v                          v
Espectro FFT          Estimacion LPC
                              |
                              v
                         F1, F2 y F3
                              |
                              v
                        Escalamiento
                              |
                              v
                    Clasificador KNN
                              |
                              v
                       Vocal detectada
```

### Componentes

- **Captura y preprocesamiento:** emplea `sounddevice`, `NumPy` y `SciPy`.
- **Visualización:** emplea `Matplotlib` para mostrar la señal y el espectro.
- **Almacenamiento:** guarda grabaciones y espectros en archivos `.npz`.
- **Análisis LPC:** calcula formantes y genera archivos `.csv`.
- **Entrenamiento:** usa `pandas`, `scikit-learn` y `joblib`.
- **Reconocimiento:** carga el modelo guardado y clasifica una grabación capturada en tiempo real.

Las carpetas se manejan con rutas relativas al directorio del proyecto, por lo que los comandos deben ejecutarse desde la carpeta raíz.

## 4. Algoritmo KNN

K-Nearest Neighbors, o K vecinos más cercanos, es un algoritmo supervisado de clasificación. Durante el entrenamiento no crea una ecuación; guarda las muestras procesadas y sus etiquetas.

En este sistema se utiliza como vector de características:

```text
[F1, F2, F3]
```

Las operaciones son las siguientes:

1. Se leen los archivos CSV de formantes.
2. Se excluyen las filas que no tengan F1, F2 y F3.
3. Cada vocal se usa como etiqueta de clase.
4. Los formantes se transforman con `StandardScaler`.
5. Para una nueva captura, se calculan F1, F2 y F3 y se aplica el mismo escalamiento.
6. El clasificador mide la cercanía entre la nueva muestra y las muestras entrenadas.
7. Se revisan los tres vecinos más cercanos.
8. Se asigna la vocal que predomina entre esos vecinos.

La distancia se calcula sobre características escaladas, de modo que un formante no tenga más peso únicamente por tener valores numéricos mayores. El modelo se guarda en `modelo_vocales_knn.pkl` junto con el escalador utilizado.

## 5. Funcionalidad completa

La versión actual ofrece las siguientes funcionalidades:

- Captura de grabaciones de las vocales `a`, `e`, `i`, `o` y `u`.
- Identificación de hablante y repetición.
- Eliminación de componentes continuas y filtrado pasaaltas.
- Normalización y validación básica de amplitud.
- Visualización de la señal original, filtrada y normalizada.
- Guardado de las grabaciones en formato `.npz`.
- Cálculo y visualización del espectro en decibelios.
- Estimación LPC y obtención de F1, F2 y F3.
- Generación de CSV individuales y estadísticos por vocal y hablante.
- Entrenamiento de un modelo KNN.
- Reconocimiento de una vocal mediante el microfono.
- Persistencia del modelo y el escalador.

## 6. Video demostrativo

**Agregar aquí el enlace al video demostrativo del proyecto:** `[Enlace pendiente]`

El video deberá mostrar:

- La instalación y preparación del entorno.
- La ejecución de la captura de una vocal.
- La visualización del espectro.
- La obtención de los formantes.
- El entrenamiento del modelo.
- Una predicción en vivo.
- Una breve explicación de los resultados.
- https://youtu.be/MA48JsqsPqA

## Requisitos

- Windows
- Python 3.10 o posterior
- Microfono
- Entorno virtual de Python recomendado

## Instalación

Desde la carpeta del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell impide activar el entorno virtual, ejecuta primero:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Comprueba que Python detecta el microfono antes de grabar. En Windows, habilita el acceso al dispositivo desde **Configuración > Privacidad y seguridad > Microfono**.

## Uso

Ejecuta los programas desde la carpeta del proyecto. Mantén la vocal sostenida durante aproximadamente 3 segundos y habla cerca del microfono.

### 1. Capturar vocales

```powershell
python 01bcapturavocales.py
```

Solicita la vocal, el identificador del hablante y el número de repetición. Las grabaciones se guardan en `capturas_vocales/` como archivos `.npz`.

### 2. Visualizar un espectro

```powershell
python 02bvocales.py
```

Solicita hablante, vocal y repetición. Los espectros en formato `.npz` se guardan en `espectros_vocales/`.

### 3. Calcular formantes

```powershell
python 03bvocales.py
```

Solicita el hablante, la vocal y la cantidad de repeticiones. Genera:

- `resultados_formantes/<HABLANTE>_<VOCAL>_formantes.csv`
- `resultados_formantes/<HABLANTE>_<VOCAL>_estadisticas.csv`

### 4. Entrenar el modelo KNN

Debes tener al menos resultados de las vocales que quieras clasificar dentro de `resultados_formantes/`.

```powershell
python 04zbknn.py
```

El modelo entrenado se guarda en `modelo_vocales_knn.pkl`.

### 5. Reconocer una vocal

```powershell
python 05brecon.py
```

El programa graba 3 segundos, calcula F1, F2 y F3 y muestra la vocal predicha.

## Estructura

```text
01bcapturavocales.py       Captura y preprocesamiento
02bvocales.py              Visualización del espectro FFT
03bvocales.py              Estimación LPC y formantes
04zbknn.py                 Entrenamiento del modelo KNN
05brecon.py                Captura y reconocimiento
modelo_vocales_knn.pkl     Modelo y escalador KNN incluidos
capturas_vocales/          Grabaciones originales procesadas
espectros_vocales/         Espectros calculados
resultados_formantes/      CSV de formantes y estadísticas
requirements.txt           Dependencias de pip
```

## Datos y convenciones

- Vocales admitidas: `a`, `e`, `i`, `o`, `u`.
- Identificador recomendado para hablantes: `H01`, `H02`, etc.
- Nombre típico de grabación: `H01_a_rep01_AAMMDD_HHMMSS.npz`.
- La extracción de formantes usa un marco de 30 ms ubicado inicialmente a 0.80 segundos.
- Los archivos existentes en las carpetas de datos forman parte del conjunto de ejemplos y pueden utilizarse para entrenar o validar el sistema.
