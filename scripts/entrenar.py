"""Entrena los modelos de ML con AneRBC-I.

Pasos:
  1. Etiquetas por imagen desde los reportes de morfología (patólogos) y el hemograma.
  2. Segmentación + medición de cada eritrocito (morfologia.py), con caché.
  3. Calibración de escala: se asume diámetro mediano de 7.5 µm en los sanos.
  4. Características agregadas por imagen -> un clasificador por hallazgo.
  5. Validación cruzada estratificada de 5 particiones (1 imagen = 1 individuo).

Uso:  python scripts/entrenar.py            (todas las imágenes)
      python scripts/entrenar.py 200        (muestra de 200 imágenes, para probar rápido)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from skimage import io
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
import morfologia  # noqa: E402
from etiquetas_anerbc import construir  # noqa: E402

DATOS = RAIZ / "datos" / "AneRBC-I"
CACHE = RAIZ / "resultados" / "cache_celulas"
OBJETIVOS = ["anemico", "microcitico", "hipocromico", "dianocitos", "eliptocitos"]
MIN_POSITIVOS = 30
DIAM_NORMAL_UM = 7.5


def procesar(args):
    grupo, ident = args
    destino = CACHE / f"{ident}.csv"
    if destino.exists():
        return ident, pd.read_csv(destino)
    img = io.imread(DATOS / grupo / "Original_images" / f"{ident}.png")
    _, df, _, _ = morfologia.analizar_celulas(img)
    df.to_csv(destino, index=False)
    return ident, df


def metricas(y, prob, umbral=0.5):
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"AUC": roc_auc_score(y, prob), "sensibilidad": tp / (tp + fn), "especificidad": tn / (tn + fp),
            "exactitud_balanceada": balanced_accuracy_score(y, pred), "positivos": int(y.sum()), "n": len(y)}


def main(limite=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    etiquetas = construir()
    if limite:
        etiquetas = etiquetas.groupby("grupo", group_keys=False).apply(
            lambda g: g.sample(min(len(g), limite // 2), random_state=0))
    tareas = list(zip(etiquetas["grupo"], etiquetas["id"]))
    print(f"Procesando {len(tareas)} imágenes...", flush=True)
    celulas = {}
    with ProcessPoolExecutor() as ex:
        for i, (ident, df) in enumerate(ex.map(procesar, tareas, chunksize=4), 1):
            celulas[ident] = df
            if i % 100 == 0:
                print(f"  {i}/{len(tareas)}", flush=True)
    if len(sys.argv) > 2 and sys.argv[2] == "solo_extraer":
        return

    # AneRBC mezcla aumentos: se agrupan las imágenes por diámetro mediano en px y se calibra
    # cada grupo asumiendo que el diámetro mediano de sus sanos es 7.5 µm.
    etiquetas["diam_px"] = [celulas[i]["diametro_px"].median() if len(celulas[i]) else np.nan for i in etiquetas["id"]]
    base = etiquetas["diam_px"].median()
    etiquetas["aumento"] = np.where(etiquetas["diam_px"] > 1.4 * base, "alto", "normal")
    print("\nImágenes por grupo de aumento:\n", pd.crosstab(etiquetas["aumento"], etiquetas["grupo"]))
    calibracion = {}
    for g, sub in etiquetas.groupby("aumento"):
        ref = sub.loc[sub.anemico == 0, "diam_px"]
        ref = ref if len(ref) >= 10 else sub["diam_px"]
        calibracion[g] = DIAM_NORMAL_UM / ref.median()
        print(f"Calibración grupo '{g}': {calibracion[g]:.3f} µm/px (diámetro de referencia {ref.median():.1f} px)")

    filas = []
    for _, fila in etiquetas.iterrows():
        r = morfologia.resumen_imagen(celulas[fila["id"]], calibracion[fila["aumento"]])
        if r:
            filas.append({"id": fila["id"], **r})
    X = pd.DataFrame(filas).merge(etiquetas, on="id")
    todas = [c for c in filas[0] if c not in ("id", "n_celulas")]   # n_celulas delata el aumento, no la patología
    de_tamano = {"diam_mediana", "diam_p10", "diam_p90", "frac_micro", "frac_macro"}
    variantes = {"completo": todas, "sin_escala": [c for c in todas if c not in de_tamano]}
    print(f"Imágenes con suficientes células: {len(X)} de {len(etiquetas)}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    modelos, reporte = {v: {} for v in variantes}, {v: {} for v in variantes}
    for variante, caracteristicas in variantes.items():
        print(f"\n===== Variante: {variante} ({len(caracteristicas)} características) =====")
        for obj in OBJETIVOS:
            y = X[obj].values
            if y.sum() < MIN_POSITIVOS or (len(y) - y.sum()) < MIN_POSITIVOS:
                print(f"[{obj}] omitido: solo {int(y.sum())} positivos")
                continue
            candidatos = {
                "regresion_logistica": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                                    class_weight="balanced", random_state=0),
            }
            res = {}
            for nombre, m in candidatos.items():
                prob = cross_val_predict(m, X[caracteristicas], y, cv=cv, method="predict_proba")[:, 1]
                res[nombre] = metricas(y, prob)
            mejor = max(res, key=lambda k: res[k]["AUC"])
            modelos[variante][obj] = candidatos[mejor].fit(X[caracteristicas], y)
            r = {"modelo": mejor, **{k: round(float(v), 3) for k, v in res[mejor].items()},
                 "comparacion_AUC": {k: round(float(v["AUC"]), 3) for k, v in res.items()}}
            reporte[variante][obj] = r
            print(f"[{obj:12}] AUC={r['AUC']:.3f} sens={r['sensibilidad']:.3f} esp={r['especificidad']:.3f} "
                  f"(positivos {r['positivos']}/{r['n']}) {mejor}; AUC por modelo {r['comparacion_AUC']}")
    um_por_px = calibracion

    # Referencia: ¿qué tan fácil es predecir "anémico" solo con la hemoglobina del hemograma?
    if "HGB" in X:
        ok = X["HGB"].notna()
        reporte["referencia_solo_HGB_AUC"] = round(float(roc_auc_score(X.loc[ok, "anemico"], -X.loc[ok, "HGB"])), 3)
        print(f"\nReferencia: AUC usando solo HGB del hemograma = {reporte['referencia_solo_HGB_AUC']}")

    (RAIZ / "modelos").mkdir(exist_ok=True)
    # Rango de cada característica en el entrenamiento: la app no predice si la imagen cae fuera
    rangos = {c: (float(X[c].quantile(0.005)), float(X[c].quantile(0.995))) for c in todas}
    joblib.dump({"modelos": modelos, "caracteristicas": variantes, "calibracion_entrenamiento": um_por_px,
                 "rangos": rangos, "metricas": reporte}, RAIZ / "modelos" / "modelo_anerbc.joblib")
    (RAIZ / "resultados" / "metricas_anerbc.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    X.to_csv(RAIZ / "resultados" / "caracteristicas_anerbc.csv", index=False)
    print("\nModelo guardado en modelos/modelo_anerbc.joblib")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
