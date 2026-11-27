"""Descarga AneRBC-I desde Mendeley (versión 1, licencia CC BY 4.0).

Uso:
    python scripts/descargar_anerbc.py reportes            # CBC + morfología (pocos KB)
    python scripts/descargar_anerbc.py imagenes            # imágenes originales (~1.7 GB)
    python scripts/descargar_anerbc.py mascaras            # segmentación binaria (~1 GB)
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/hms3sjzt7f/files?folder_id={}&version=1"
DESTINO = Path(__file__).resolve().parent.parent / "datos" / "AneRBC-I"

CARPETAS = {
    "reportes": {
        ("Anemic", "CBC_reports"): "17cbb7ca-49f0-4ea4-99c9-489508b36c6e",
        ("Anemic", "Morphology_reports"): "cc2168b3-3350-4d01-a5b9-a906d35e5617",
        ("Healthy", "CBC_reports"): "8f5b2fc4-7d1e-4992-b3b4-2ab82663ccc3",
        ("Healthy", "Morphology_reports"): "884735f2-f3e7-48f2-815f-0645b1950074",
    },
    "imagenes": {
        ("Anemic", "Original_images"): "65735c00-0a3a-4a4b-9259-0ac5bbd8e662",
        ("Healthy", "Original_images"): "a7146be7-b2f4-4a4a-a25d-679bec86b1cd",
    },
    "mascaras": {
        ("Anemic", "Binary_segmented"): "f1e22862-5e78-4276-a178-f86318748385",
        ("Healthy", "Binary_segmented"): "85d22af6-4eb6-4df8-9955-14da6a8820e4",
    },
}


def abrir(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    return urllib.request.urlopen(req, timeout=timeout)


def bajar(item, carpeta):
    destino = carpeta / Path(item["filename"]).name          # solo el nombre, sin rutas
    if destino.exists() and destino.stat().st_size == item["size"]:
        return 0
    for intento in range(3):
        try:
            with abrir(item["content_details"]["download_url"], 180) as r:
                destino.write_bytes(r.read())
            return 1
        except Exception:
            if intento == 2:
                raise


def main(tipo):
    for (grupo, sub), fid in CARPETAS[tipo].items():
        with abrir(API.format(fid), 60) as r:
            items = json.load(r)
        carpeta = DESTINO / grupo / sub
        carpeta.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(8) as ex:
            nuevos = sum(ex.map(lambda it: bajar(it, carpeta), items))
        print(f"{grupo}/{sub}: {len(items)} archivos ({nuevos} nuevos)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reportes")
