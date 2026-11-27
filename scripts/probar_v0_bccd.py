"""Prueba la v0 con fotos reales de BCCD y compara el conteo contra las anotaciones (recuadros RBC)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from skimage import io

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import app  # noqa: E402

bccd = next((RAIZ / "datos" / "BCCD").glob("BCCD_Dataset-*")) / "BCCD"
salida = RAIZ / "resultados" / "v0_bccd"
salida.mkdir(parents=True, exist_ok=True)

imagenes = sorted((bccd / "JPEGImages").glob("*.jpg"))[:: max(1, len(list((bccd / "JPEGImages").glob("*.jpg"))) // 40)]
filas = []
for i, ruta in enumerate(imagenes):
    img = io.imread(ruta)
    anot = ET.parse(bccd / "Annotations" / (ruta.stem + ".xml")).getroot()
    n_rbc = sum(1 for o in anot.iter("object") if o.findtext("name") == "RBC")
    out, _, df = app.analizar(img, 0)
    n = len(df)
    filas.append((ruta.name, n_rbc, n, df["diámetro"].median() if n else np.nan))
    if i < 6:
        io.imsave(salida / ruta.name.replace(".jpg", "_v0.png"), out, check_contrast=False)

print(f"{'imagen':22} {'RBC anotados':>12} {'v0 detecta':>10} {'diám. mediano px':>16}")
for f in filas:
    print(f"{f[0]:22} {f[1]:>12} {f[2]:>10} {f[3]:>16.1f}")
a = np.array([f[1] for f in filas]); d = np.array([f[2] for f in filas])
print(f"\nImágenes: {len(filas)} | anotados promedio {a.mean():.1f} | detectados promedio {d.mean():.1f}")
print("Nota: BCCD tiene anotaciones incompletas y la v0 excluye células cortadas por el borde.")
