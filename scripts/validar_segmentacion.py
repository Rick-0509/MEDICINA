"""Valida la segmentación clásica (morfologia.py) en AneRBC:
  1. contra las máscaras binarias de los autores (IoU de píxeles),
  2. contra Cellpose: ¿las medidas agregadas por imagen coinciden? (la app usa Cellpose).
Uso: python scripts/validar_segmentacion.py [n_imagenes_cellpose]
"""
import sys
from pathlib import Path

import numpy as np
from skimage import io

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import morfologia  # noqa: E402

DATOS = RAIZ / "datos" / "AneRBC-I"
n_cp = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rng = np.random.default_rng(0)
CLAVES = ["n_celulas", "diam_mediana", "diam_cv", "palidez_mediana", "elong_mediana", "frac_alargadas", "densidad_mediana"]

print("1) Clásica vs máscaras de los autores")
ious = []
muestra = []
for grupo in ["Anemic", "Healthy"]:
    rutas = sorted((DATOS / grupo / "Binary_segmented").glob("*.png"))
    for ruta in rng.choice(rutas, size=6, replace=False):
        img = io.imread(DATOS / grupo / "Original_images" / ruta.name)[..., :3]
        ref = io.imread(ruta)
        ref = (ref[..., 0] if ref.ndim == 3 else ref) <= 127          # células en negro
        etiquetas, teñido, _ = morfologia.segmentar(img)
        nuestra = morfologia.ndi.binary_fill_holes(teñido)
        m = np.zeros_like(ref); m[40:-40, 40:-40] = True
        ious.append((ref & nuestra & m).sum() / ((ref | nuestra) & m).sum())
        muestra.append((grupo, ruta.name))
print(f"   IoU promedio de píxeles (12 imágenes): {np.mean(ious):.3f}  (mín {np.min(ious):.3f})")

print(f"\n2) Clásica vs Cellpose en {n_cp} imágenes (medidas en px, sin calibrar)")
from cellpose import models  # noqa: E402
cp = models.Cellpose(gpu=False, model_type="cyto3")
print(f"   {'imagen':10} " + " ".join(f"{k:>16}" for k in CLAVES))
for grupo, nombre in muestra[:n_cp // 2] + muestra[6:6 + n_cp // 2]:
    img = io.imread(DATOS / grupo / "Original_images" / nombre)[..., :3]
    for metodo in ["clasica", "cellpose"]:
        if metodo == "clasica":
            _, df, _, _ = morfologia.analizar_celulas(img)
        else:
            _, df, _, _ = morfologia.analizar_celulas(img, segmentador=cp)
        r = morfologia.resumen_imagen(df, 1.0)
        print(f"   {nombre[:7]:7}{metodo[:3]} " + " ".join(f"{r[k]:>16.3f}" for k in CLAVES), flush=True)
