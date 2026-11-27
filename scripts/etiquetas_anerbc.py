"""Convierte los reportes de morfología y hemograma de AneRBC-I en una tabla de etiquetas por imagen."""
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos" / "AneRBC-I"

# Hallazgos que se buscan en el texto del reporte (en minúsculas)
PATRONES = {
    "microcitico": r"microcyt",
    "macrocitico": r"macrocyt",
    "hipocromico": r"hypochrom",
    "dianocitos": r"target cell",
    "eliptocitos": r"elliptocyt|ovalocyt|pencil cell",
    "anisocitosis": r"anisocyt",
    "poiquilocitosis": r"poikilocyt",
}
CBC = ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "RDW"]


def leer_cbc(ruta):
    vals = {}
    for linea in ruta.read_text(encoding="latin-1").splitlines()[1:]:
        partes = linea.split(",")
        if len(partes) >= 2 and partes[0].strip() in CBC:
            m = re.search(r"\d+(?:\.\d+)?", partes[1])
            if m:
                vals[partes[0].strip()] = float(m.group())
    return vals


def construir():
    filas = []
    for grupo, anemico in [("Anemic", 1), ("Healthy", 0)]:
        for ruta in sorted((DATOS / grupo / "Morphology_reports").glob("*.txt")):
            texto = ruta.read_text(encoding="latin-1")
            m = re.search(r"PERIPHERAL SMEAR:(.*?)(WBCs MORPHOLOGY|$)", texto, re.S | re.I)
            rbc = " ".join((m.group(1) if m else texto).split()).lower()
            fila = {"id": ruta.stem, "grupo": grupo, "anemico": anemico, "reporte_rbc": rbc}
            for k, p in PATRONES.items():
                fila[k] = int(bool(re.search(p, rbc)))
            cbc = DATOS / grupo / "CBC_reports" / ruta.name
            if cbc.exists():
                fila.update(leer_cbc(cbc))
            filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = construir()
    df.to_csv(RAIZ / "datos" / "etiquetas_anerbc.csv", index=False)
    print(df.groupby("grupo")[list(PATRONES)].sum().T)
    print("\nEjemplos de reportes (anémicos):")
    for t in df[df.anemico == 1]["reporte_rbc"].sample(6, random_state=1):
        print(" -", t[:160])
    print("\nHGB promedio por grupo:\n", df.groupby("grupo")[["HGB", "MCV", "MCH"]].mean().round(1))
