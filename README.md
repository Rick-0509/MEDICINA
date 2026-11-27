# Morfología de eritrocitos — prototipo v1

Plataforma web que analiza fotos de frotis de sangre periférica: separa cada glóbulo rojo, mide su tamaño, forma y palidez central, y estima hallazgos frecuentes en anemia con un modelo de machine learning.

> **Prototipo de investigación. No es un dispositivo de diagnóstico.** El modelo solo está validado con el dataset AneRBC. Todo resultado requiere confirmación por un tecnólogo médico.

## Cómo funciona

1. **Segmentación:** Cellpose (red neuronal ya entrenada) o un método clásico separan cada glóbulo rojo.
2. **Medición:** tamaño, palidez central, forma y color de cada célula.
3. **Resumen:** 15 medidas por muestra.
4. **Modelo:** Gradient Boosting (scikit-learn) estima la probabilidad de anemia, microcitosis, hipocromía, dianocitos y eliptocitos. Si la foto no se parece a las de entrenamiento, no predice.
5. **Interpretación:** lectura orientativa en lenguaje sencillo, con los exámenes que conviene pedir.

Desempeño en AneRBC (validación cruzada de 5 particiones, AUC): anemia 0,94 · microcitosis 0,89 · hipocromía 0,88 · eliptocitos 0,85 · dianocitos 0,83.

## Instalación (Windows)

Requiere **Python 3.12** ([descarga](https://www.python.org/downloads/release/python-31210/)). No usar 3.13 ni superior: `numpy 2.0.2`, que exige Cellpose, no se instala en esas versiones. La instalación descarga unos 1,5 GB (PyTorch).

```powershell
git clone https://github.com/Rick-0509/MEDICINA.git
cd MEDICINA
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Las versiones de `requirements.txt` están fijadas a propósito: el modelo se guardó con `scikit-learn 1.9.1` y puede no cargar con otra versión.

**Clona el repositorio en una ruta corta** (por ejemplo `C:\proyectos\MEDICINA` o el Escritorio). PyTorch incluye archivos con nombres muy largos y Windows limita las rutas a 260 caracteres: en carpetas muy anidadas la instalación falla con `OSError: [Errno 2] No such file or directory`. Otra solución es [habilitar las rutas largas de Windows](https://pip.pypa.io/warnings/enable-long-paths).

Probado en Windows 11 con Python 3.12. Requiere unos 2 GB de RAM libres e internet para la instalación y para el primer uso de Cellpose.

## Uso

```powershell
.\venv\Scripts\python.exe app.py
```

Abrir `http://127.0.0.1:7860` en el navegador. La primera vez que se usa Cellpose descarga su modelo (unos 25 MB).

El modelo entrenado ya viene incluido en `modelos/modelo_anerbc.joblib`, así que no hace falta descargar datos para usar la app.

## Reentrenar el modelo

```powershell
.\venv\Scripts\python.exe scripts\descargar_anerbc.py reportes
.\venv\Scripts\python.exe scripts\descargar_anerbc.py imagenes
.\venv\Scripts\python.exe scripts\entrenar.py
```

Las imágenes pesan unos 1,7 GB. La primera extracción de medidas tarda alrededor de 45 minutos; después se usa la caché.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `app.py` | Interfaz web (Gradio), análisis e interpretación |
| `morfologia.py` | Segmentación y medición de cada célula |
| `scripts/` | Descarga de datos, etiquetas, entrenamiento y pruebas |
| `modelos/` | Modelo entrenado |
| `resultados/` | Métricas y características por imagen |

## Datos y licencia

El modelo se entrenó con **AneRBC-I** (licencia CC BY 4.0):

> Shahzad M. et al. *AneRBC dataset: a benchmark dataset for computer-aided anemia diagnosis using RBC images.* Database, 2024, baae120. https://doi.org/10.1093/database/baae120
> Datos: https://data.mendeley.com/datasets/hms3sjzt7f/1

Las pruebas con fotos a alto aumento usaron [BCCD](https://github.com/Shenggan/BCCD_Dataset).

## Privacidad

Las fotos de pacientes son datos de salud sensibles (Ley 29733). No subir imágenes de pacientes a este repositorio y usar la plataforma localmente o en un entorno privado.
