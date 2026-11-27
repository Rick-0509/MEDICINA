"""Control de sesgo: ¿el modelo aprende la patología o el aumento de la foto?
Repite la validación cruzada dentro de un solo grupo de aumento y predice el aumento con las características."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RAIZ = Path(__file__).resolve().parent.parent
X = pd.read_csv(RAIZ / "resultados" / "caracteristicas_anerbc.csv")
TAMANO = {"diam_mediana", "diam_p10", "diam_p90", "frac_micro", "frac_macro"}
todas = ["diam_mediana", "diam_p10", "diam_p90", "diam_cv", "frac_micro", "frac_macro", "palidez_mediana",
         "palidez_p90", "frac_palidez_alta", "elong_mediana", "frac_alargadas", "frac_irregulares",
         "densidad_mediana", "saturacion_mediana", "frac_centro_oscuro"]
variantes = {"completo": todas, "sin_escala": [c for c in todas if c not in TAMANO]}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
gb = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=0)

sub = X[X.aumento == "normal"]
print(f"Solo aumento 'normal' ({len(sub)} imágenes):")
for v, cols in variantes.items():
    res = []
    for obj in ["anemico", "microcitico", "hipocromico", "dianocitos", "eliptocitos"]:
        p = cross_val_predict(gb(), sub[cols], sub[obj], cv=cv, method="predict_proba")[:, 1]
        res.append(f"{obj}={roc_auc_score(sub[obj], p):.3f}")
    print(f"  {v:10}: " + "  ".join(res))

y = (X.aumento == "alto").astype(int)
p = cross_val_predict(gb(), X[variantes["sin_escala"]], y, cv=cv, method="predict_proba")[:, 1]
print(f"\n¿Se puede adivinar el aumento con las características sin escala? AUC={roc_auc_score(y, p):.3f}")

# ¿Los hallazgos se predicen por sí mismos o solo porque van juntos con 'anémico'?
print("\nDentro de los anémicos (¿distingue hallazgos entre pacientes anémicos?):")
an = X[X.anemico == 1]
for obj in ["dianocitos", "eliptocitos"]:
    if 30 <= an[obj].sum() <= len(an) - 30:
        p = cross_val_predict(gb(), an[todas], an[obj], cv=cv, method="predict_proba")[:, 1]
        print(f"  {obj}: AUC={roc_auc_score(an[obj], p):.3f} (positivos {int(an[obj].sum())}/{len(an)})")
