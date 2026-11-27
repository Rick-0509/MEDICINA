"""Prueba de punta a punta de app.analizar con fotos reales (BCCD con Cellpose, AneRBC con clásico)."""
import sys
from pathlib import Path

import numpy as np
from skimage import io

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import app  # noqa: E402

salida = RAIZ / "resultados" / "app_v1"
salida.mkdir(parents=True, exist_ok=True)
bccd = RAIZ / "datos/BCCD/BCCD_Dataset-master/BCCD/JPEGImages"
campos_bccd = [str(bccd / f"BloodImage_{i:05d}.jpg") for i in range(0, 60, 4)]   # 15 campos
casos = [("BCCD, 15 campos", campos_bccd, "Cellpose"),
         ("AneRBC 001_a (anémico)", [str(RAIZ / "datos/AneRBC-I/Anemic/Original_images/001_a.png")], "Clásico"),
         ("AneRBC 001_h (sano)", [str(RAIZ / "datos/AneRBC-I/Healthy/Original_images/001_h.png")], "Clásico")]
for nombre, archivos, metodo in casos:
    galeria, reporte, tabla = app.analizar(archivos, 0, metodo)
    io.imsave(salida / (nombre.split(",")[0].replace(" ", "_") + ".png"), galeria[0][0], check_contrast=False)
    print(f"\n######## {nombre} ({metodo})")
    print("\n".join(l for l in reporte.splitlines() if l.strip() and not l.startswith("⚠️ **Prototipo")))
