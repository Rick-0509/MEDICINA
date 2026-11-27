# Prototipo v1: medición morfológica de eritrocitos + modelo de ML entrenado con AneRBC.
# USO DE INVESTIGACIÓN. No es un dispositivo de diagnóstico.
import html
import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io, segmentation, util

import morfologia

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "modelos" / "modelo_anerbc.joblib"

# Umbrales de categoría por célula (a validar con el tecnólogo médico)
MICRO_UM, MACRO_UM = 6.0, 9.0
PALIDEZ_MAX = 0.5      # palidez central > 50% del diámetro
ELONG_MIN = 0.7        # eje menor / eje mayor
SOLIDEZ_MIN = 0.9      # por debajo, la forma no es convexa: casi siempre es una célula superpuesta

COLORES = {"superpuesta": (0, 0.8, 0.8), "microcito": (0, 0, 1), "macrocito": (1, 0, 0), "alargada": (1, 0, 1),
           "hipocrómica": (1, 0.8, 0), "sin alteración": (0, 0.8, 0)}
NOMBRES = {"anemico": "Anemia (patrón global)", "microcitico": "Microcitosis", "hipocromico": "Hipocromía",
           "dianocitos": "Dianocitos (target cells)", "eliptocitos": "Eliptocitos"}
METODO_CELLPOSE = "Cellpose (recomendado, ~15-60 s por foto)"
METODO_CLASICO = "Clásico (rápido, solo monocapa bien separada)"

_cellpose = None
_modelo = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else None


# ----------------------------------------------------------------------------------------------
# Análisis (sin cambios de lógica)
# ----------------------------------------------------------------------------------------------
def cellpose():
    global _cellpose
    if _cellpose is None:
        from cellpose import models
        _cellpose = models.Cellpose(gpu=False, model_type="cyto3")
    return _cellpose


def categoria(fila, um):
    if fila["solidez"] < SOLIDEZ_MIN:      # tapada por otra célula: su tamaño y forma no son confiables
        return "superpuesta"
    if um and fila["diam_um"] < MICRO_UM:
        return "microcito"
    if um and fila["diam_um"] > MACRO_UM:
        return "macrocito"
    if fila["elongacion"] < ELONG_MIN:
        return "alargada"
    if fila["palidez"] > PALIDEZ_MAX:
        return "hipocrómica"
    return "sin alteración"


def predecir(df, um):
    """Probabilidades del modelo de ML. Sin calibración usa la variante que no depende del tamaño."""
    if _modelo is None:
        return None, "Modelo no entrenado todavía (ejecuta scripts/entrenar.py)."
    variante = "completo" if um else "sin_escala"
    resumen = morfologia.resumen_imagen(df, um or 1.0)
    if resumen is None:
        return None, "Muy pocas células para el modelo (mínimo 20)."
    cols = _modelo["caracteristicas"][variante]
    # Control de dominio: si la imagen cae fuera del rango visto en el entrenamiento, no se predice.
    fuera = [c for c in cols if not (_modelo["rangos"][c][0] <= resumen[c] <= _modelo["rangos"][c][1])]
    if len(fuera) >= 2:
        return None, ("**No se muestran predicciones:** la imagen está fuera del rango de las imágenes de entrenamiento "
                      f"en {len(fuera)} medidas ({', '.join(fuera)}). El modelo no es confiable para este tipo de foto "
                      "(aumento, tinción o superposición distintos). Hace falta entrenarlo con fotos de su laboratorio. "
                      "Las mediciones por célula y los contornos sí siguen siendo válidos.")
    X = pd.DataFrame([resumen])[cols]
    probs = {obj: float(m.predict_proba(X)[0, 1]) for obj, m in _modelo["modelos"][variante].items()}
    return probs, variante


def _leer(archivo):
    ruta = archivo if isinstance(archivo, str) else getattr(archivo, "name", archivo)
    return io.imread(ruta)


GRIS_EXCLUIDA = (0.55, 0.55, 0.55)


def dibujar(img, etiquetas, df):
    """Contorno grueso por categoría; en gris, los objetos descartados (leucocitos, plaquetas, grumos)."""
    salida = util.img_as_float(np.asarray(img)[..., :3]).copy()
    grosor = int(np.clip(round(df["diametro_px"].median() / 40), 1, 4))     # proporcional al tamaño de la célula
    excluida = (etiquetas > 0) & ~np.isin(etiquetas, df["id"].to_numpy())
    capas = [(GRIS_EXCLUIDA, np.where(excluida, etiquetas, 0))]
    for cat, col in COLORES.items():
        ids = df.loc[df["categoria"] == cat, "id"].tolist()
        capas.append((col, np.where(np.isin(etiquetas, ids), etiquetas, 0)))
    for col, lab in capas:
        if lab.any():
            borde = segmentation.find_boundaries(lab, mode="inner")
            salida[ndi.binary_dilation(borde, iterations=grosor - 1) if grosor > 1 else borde] = col
    return (salida * 255).astype(np.uint8)


def procesar(archivos, um_por_px, metodo, progreso=None):
    """Analiza uno o varios campos del MISMO frotis y agrega todas las células."""
    if not archivos:
        return {"ok": False, "msg": "Sube al menos una imagen."}
    archivos = archivos if isinstance(archivos, list) else [archivos]
    um = float(um_por_px) if um_por_px and um_por_px > 0 else None
    seg = cellpose() if metodo.startswith("Cellpose") else None

    partes, imagenes, excluidas = [], [], 0
    for n, archivo in enumerate(archivos, 1):
        if progreso:
            progreso((n - 1) / len(archivos), desc=f"Analizando campo {n} de {len(archivos)}…")
        img = _leer(archivo) if not isinstance(archivo, np.ndarray) else archivo
        img = np.asarray(img)[..., :3] if np.asarray(img).ndim == 3 else img
        etiquetas, df_i, exc, _ = morfologia.analizar_celulas(img, segmentador=seg)
        excluidas += exc
        if df_i.empty:
            continue
        df_i["campo"] = n
        df_i["categoria"] = [categoria({**f, "diam_um": f["diametro_px"] * (um or np.nan)}, um) for _, f in df_i.iterrows()]
        imagenes.append((dibujar(img, etiquetas, df_i), f"Campo {n}: {len(df_i)} eritrocitos"))
        partes.append(df_i)
    if not partes:
        return {"ok": False, "msg": "No se detectaron eritrocitos válidos. Revisa enfoque, iluminación y que sea zona de monocapa."}

    df = pd.concat(partes, ignore_index=True)
    df["diam_um"] = df["diametro_px"] * (um or np.nan)
    if progreso:
        progreso(0.95, desc="Aplicando el modelo…")
    probs, estado = predecir(df, um)
    return {"ok": True, "imagenes": imagenes, "df": df, "excluidas": excluidas, "um": um,
            "campos": len(partes), "probs": probs, "estado": estado}


def _estadisticas(df, um):
    d = df["diam_um"] if um else df["diametro_px"]
    s = {"d_med": float(d.median()), "unidad": "µm" if um else "px", "d_px": float(df["diametro_px"].median()),
         "cv": float(100 * d.std() / d.mean()),
         "pal_med": float(df["palidez"].median()), "pal_p90": float(df["palidez"].quantile(.9)),
         "pal_alta": float(100 * (df["palidez"] > PALIDEZ_MAX).mean()),
         "alarg": float(100 * ((df["elongacion"] < ELONG_MIN) & (df["solidez"] >= SOLIDEZ_MIN)).mean()),
         "centro": float(100 * (df["centro_vs_medio"] > 0.03).mean())}
    if um:
        s["micro"] = float(100 * (df["diam_um"] < MICRO_UM).mean())
        s["macro"] = float(100 * (df["diam_um"] > MACRO_UM).mean())
    return s


def _tabla(df):
    return df[["campo", "id", "diametro_px", "diam_um", "palidez", "elongacion", "centro_vs_medio", "categoria"]].round(3)


AVISO_VALIDACION = ("Prototipo de investigación. El modelo solo se validó con imágenes de AneRBC (un hospital, una tinción, "
                    "aumento bajo). Con fotos de otro laboratorio o de celular su desempeño no está validado. "
                    "Requiere confirmación por tecnólogo médico; la morfología no reemplaza ferritina ni electroforesis.")


def analizar(archivos, um_por_px, metodo):
    """Versión en texto (Markdown). La usan los scripts de prueba."""
    r = procesar(archivos, um_por_px, metodo)
    if not r["ok"]:
        return [], r["msg"], pd.DataFrame()
    df, um = r["df"], r["um"]
    s = _estadisticas(df, um)
    rep = [f"### Mediciones ({len(df)} eritrocitos en {r['campos']} campo(s), {r['excluidas']} objetos excluidos)",
           f"- Diámetro mediano: **{s['d_med']:.2f} {s['unidad']}** · Anisocitosis (CV del diámetro): **{s['cv']:.1f}%**",
           f"- Índice de palidez central (mediana / p90): **{s['pal_med']:.2f} / {s['pal_p90']:.2f}** "
           "*(medida relativa; el corte clínico aún no está calibrado)*",
           f"- Formas alargadas: **{s['alarg']:.1f}%** · Centro oscuro (posible dianocito): **{s['centro']:.1f}%**"]
    if um:
        rep.append(f"- Microcitos (<{MICRO_UM:g} µm): **{s['micro']:.1f}%** · Macrocitos (>{MACRO_UM:g} µm): **{s['macro']:.1f}%**")
    else:
        rep.append("- Sin calibración: no se clasifican micro/macrocitos.")
    rep.append("\n### Modelo de machine learning (entrenado con AneRBC)")
    if r["probs"] is None:
        rep.append(f"- {r['estado']}")
    else:
        for obj, p in r["probs"].items():
            rep.append(f"- {NOMBRES.get(obj, obj)}: **{100 * p:.0f}%** de probabilidad")
        rep.append(f"\n*Variante usada: {'con tamaño (calibrada)' if r['estado'] == 'completo' else 'sin tamaño (no calibrada)'}.*")
    if len(df) < 200:
        rep.append(f"\n⚠️ Solo {len(df)} células: el modelo se entrenó con ~1.000 células por muestra. "
                   "Sube más campos del mismo frotis (a 100x suelen hacer falta 10 o más).")
    rep.append(f"\n⚠️ **{AVISO_VALIDACION}**")
    return r["imagenes"], "\n".join(rep), _tabla(df)


# ----------------------------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------------------------
def _md(texto):
    """Escapa HTML y convierte **negritas** a <b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(texto))


def _svg_celulas():
    """Frotis ilustrado: eritrocitos bicóncavos, un dianocito y un eliptocito."""
    # (x, y, radio, opacidad, tipo)
    celulas = [(430, 40, 46, .95, "n"), (520, 118, 38, .9, "n"), (345, 128, 30, .8, "n"), (590, 38, 30, .85, "d"),
               (462, 168, 34, .75, "n"), (650, 100, 40, .9, "n"), (555, 175, 26, .8, "e"), (700, 185, 30, .7, "n")]
    partes = []
    for x, y, r, op, tipo in celulas:
        if tipo == "e":
            partes.append(f'<g transform="rotate(-28 {x} {y})" opacity="{op}"><ellipse cx="{x}" cy="{y}" rx="{r * 1.7:.0f}" '
                          f'ry="{r * .8:.0f}" fill="url(#m-rbc)"/><ellipse cx="{x}" cy="{y}" rx="{r * .7:.0f}" ry="{r * .26:.0f}" '
                          f'fill="#ffd3d6" opacity=".3"/></g>')
            continue
        partes.append(f'<g opacity="{op}"><circle cx="{x}" cy="{y}" r="{r}" fill="url(#m-rbc)"/>'
                      f'<circle cx="{x}" cy="{y}" r="{r * .4:.1f}" fill="#ffd3d6" opacity=".32"/>'
                      + (f'<circle cx="{x}" cy="{y}" r="{r * .16:.1f}" fill="#7d1426" opacity=".85"/>' if tipo == "d" else "")
                      + "</g>")
    return ('<svg viewBox="0 0 760 220" preserveAspectRatio="xMaxYMid slice" aria-hidden="true">'
            '<defs><radialGradient id="m-rbc"><stop offset="0" stop-color="#f8c4c8"/><stop offset=".3" stop-color="#e9747e"/>'
            '<stop offset=".78" stop-color="#c8243b"/><stop offset="1" stop-color="#8a182b"/></radialGradient></defs>'
            + "".join(partes) + "</svg>")


HERO = f"""
<div class="m-hero">
  {_svg_celulas()}
  <div class="m-hero-txt">
    <div class="m-chips"><span>Hematología digital</span><span>Prototipo v1</span><span>Uso de investigación</span></div>
    <h1>Morfología de eritrocitos</h1>
    <p>Sube fotos de un frotis de sangre periférica. La plataforma separa cada glóbulo rojo, mide su tamaño, forma y
       palidez central, y un modelo entrenado con AneRBC estima los hallazgos más frecuentes en anemia.</p>
  </div>
</div>
"""

LEYENDA = ('<div class="m-leyenda"><b>Contornos:</b><span><i style="background:#00cccc"></i>superpuesta</span>'
           '<span><i style="background:#2f6bff"></i>microcito</span><span><i style="background:#ff2f2f"></i>macrocito</span>'
           '<span><i style="background:#ff2fd0"></i>alargada</span><span><i style="background:#ffc800"></i>hipocrómica</span>'
           '<span><i style="background:#00c853"></i>sin alteración</span>'
           '<span><i style="background:#8c8c8c"></i>excluida (leucocito, plaqueta, grumo)</span></div>')

VACIO = f"""
<div class="m-vacio">
  <svg viewBox="0 0 120 70" aria-hidden="true"><defs><radialGradient id="m-rbc2"><stop offset="0" stop-color="#f8c4c8"/>
  <stop offset=".3" stop-color="#e9747e"/><stop offset=".78" stop-color="#c8243b"/><stop offset="1" stop-color="#8a182b"/>
  </radialGradient></defs><circle cx="30" cy="38" r="22" fill="url(#m-rbc2)" opacity=".9"/><circle cx="68" cy="26" r="17" fill="url(#m-rbc2)" opacity=".7"/>
  <circle cx="92" cy="48" r="14" fill="url(#m-rbc2)" opacity=".55"/><circle cx="30" cy="38" r="9" fill="#ffd3d6" opacity=".35"/>
  <circle cx="68" cy="26" r="7" fill="#ffd3d6" opacity=".35"/></svg>
  <h3>Aún no hay resultados</h3>
  <p>Sube una o varias fotos del mismo frotis y pulsa <b>Analizar frotis</b>.<br>
  También puedes probar con los ejemplos de AneRBC que aparecen abajo.</p>
</div>
"""

PIE = ("<div class='m-pie'>Las imágenes se procesan en este equipo. Si trabajas con fotos de pacientes, hazlo localmente o en un "
       "espacio privado y sin nombres en los archivos (Ley 29733). Herramienta de investigación: no es un dispositivo de diagnóstico.</div>")


def _aviso(texto, tipo="info"):
    return f'<div class="m-aviso m-aviso-{tipo}">{_md(texto)}</div>'


def _tile(valor, etiqueta, nota=""):
    nota = f'<div class="m-nota">{html.escape(nota)}</div>' if nota else ""
    return f'<div class="m-tile"><div class="m-val">{valor}</div><div class="m-lab">{etiqueta}</div>{nota}</div>'


def _barra(nombre, p):
    pct = round(100 * p)
    clase = "baja" if p < 0.35 else "media" if p < 0.65 else "alta"
    return (f'<div class="m-fila"><div class="m-nom">{html.escape(nombre)}</div>'
            f'<div class="m-pista"><div class="m-barra {clase}" style="width:{max(pct, 2)}%"></div></div>'
            f'<div class="m-pct">{pct}%</div></div>')


ALTA, BAJA = 0.65, 0.35        # mismos cortes que el color de las barras


def _lista(items):
    return "<ul>" + "".join(f"<li>{_md(t)}</li>" for t in items) + "</ul>"


def _interpretacion(r, s):
    """Lectura en lenguaje sencillo: qué se observó, por qué tomarlo con cautela y qué hacer. No es un diagnóstico."""
    um, n, probs = r["um"], len(r["df"]), r["probs"]
    motivos, pasos, patron_hipo = [], [], False

    if probs is not None:                                   # el modelo sí aplica a esta foto
        altos = {o: p for o, p in probs.items() if p >= ALTA}
        if altos:
            badge, clase = "Patrón a confirmar", "med"
            lead = ("El modelo estima **probabilidad alta** de: "
                    + ", ".join(f"{NOMBRES.get(o, o).lower()} ({p:.0%})" for o, p in altos.items())
                    + ". Es un patrón morfológico de la muestra, **no un diagnóstico**.")
        elif all(p <= BAJA for p in probs.values()):
            badge, clase = "Sin patrón marcado", "ok"
            lead = "El modelo **no encuentra un patrón marcado** de anemia ni de los hallazgos que evalúa. Eso tampoco descarta una enfermedad."
        else:
            badge, clase = "Indeterminado", "no"
            lead = "El modelo da **probabilidades intermedias**: no se inclina con claridad hacia ningún hallazgo."
        patron_hipo = probs.get("microcitico", 0) >= ALTA or probs.get("hipocromico", 0) >= ALTA
        if n < 200:
            motivos.append(f"Hay solo {n} células; el modelo se entrenó con ~1.000 por muestra.")
        motivos.append("El modelo solo se validó con imágenes de AneRBC (un hospital, una tinción); con otras fotos no está probado.")
    elif s["d_px"] >= 40 and n >= 30:                      # resolución suficiente: lectura por mediciones
        marcas = []
        if s["pal_alta"] >= 25:
            marcas.append(f"palidez central marcada en **{s['pal_alta']:.0f}%** de las células (posible hipocromía)")
        if s["alarg"] >= 20:
            marcas.append(f"**{s['alarg']:.0f}%** de formas alargadas (posibles eliptocitos)")
        if um and s["micro"] >= 30:
            marcas.append(f"**{s['micro']:.0f}%** de microcitos")
        if um and s["macro"] >= 30:
            marcas.append(f"**{s['macro']:.0f}%** de macrocitos")
        if marcas:
            badge, clase = "Hallazgos a confirmar", "med"
            lead = "Por medición se observa: " + "; ".join(marcas) + ". Es una lectura por reglas provisionales, **no un diagnóstico**."
            patron_hipo = s["pal_alta"] >= 25 or bool(um and s["micro"] >= 30)
        else:
            badge, clase = "Sin hallazgo marcado", "ok"
            lead = (f"Por medición no se marca ningún hallazgo: palidez central marcada en {s['pal_alta']:.0f}% y formas alargadas en "
                    f"{s['alarg']:.0f}% de las células. Con reglas provisionales, **esto no descarta una enfermedad**.")
        motivos.append("El modelo de machine learning no se aplica a este tipo de foto, así que la lectura se basa solo en las mediciones.")
        if n < 200:
            motivos.append(f"Hay solo {n} células; conviene analizar varios campos del mismo frotis.")
    else:                                                   # la foto no alcanza para concluir
        badge, clase = "No concluyente", "no"
        lead = ("Con esta foto **no se puede sacar una conclusión**: no es que los glóbulos rojos estén bien ni mal, "
                "es que la imagen no alcanza para evaluarlos.")
        if s["d_px"] < 40:
            motivos.append(f"La resolución es baja: cada célula ocupa solo ~{s['d_px']:.0f} px. Con tan pocos píxeles no se mide con "
                           "fiabilidad la palidez central ni la forma (un 0% de palidez aquí significa «no medible», no «normal»).")
        if n < 100:
            motivos.append(f"Hay solo {n} células analizadas.")
        motivos.append("El modelo de machine learning no se aplica a este tipo de foto (tinción, aumento o superposición distintos a los de entrenamiento).")

    pasos.append("Confirma con un **hemograma completo** (hemoglobina, VCM, HCM, ADE) y la lectura del frotis por un tecnólogo médico o hematólogo.")
    if patron_hipo:
        pasos.append("Si el patrón es microcítico e hipocrómico, lo habitual es pedir **ferritina o perfil de hierro** y, según el caso, "
                     "**electroforesis de hemoglobina**: la deficiencia de hierro y la talasemia se ven parecidas en el frotis y solo esos exámenes las distinguen.")
    if clase == "no":
        pasos.append("Mejora la foto: objetivo 100x, zona de monocapa, bien enfocada, 10 o más campos del mismo frotis, y calibra los µm por píxel.")
    pasos.append("Esta herramienta no reemplaza la consulta médica. Si es tu propia muestra y tienes síntomas "
                 "(cansancio, palidez, falta de aire, mareos), consulta a un médico.")

    por_que = f'<div class="m-sub">Por qué tomarlo con cautela</div>{_lista(motivos)}' if motivos else ""
    return ('<div class="m-card m-interp">'
            f'<div class="m-card-t">Interpretación orientativa<span class="m-badge m-b-{clase}">{badge}</span></div>'
            f'<p class="m-lead">{_md(lead)}</p>{por_que}<div class="m-sub">Qué hacer</div>{_lista(pasos)}</div>')


def _html_resultados(r):
    df, um = r["df"], r["um"]
    s = _estadisticas(df, um)
    miles = f"{len(df):,}".replace(",", ".")
    tiles = [_tile(miles, "eritrocitos", f"{r['campos']} campo(s) · {r['excluidas']} objetos excluidos"),
             _tile(f"{s['d_med']:.1f} <small>{s['unidad']}</small>", "diámetro mediano"),
             _tile(f"{s['cv']:.0f}%", "anisocitosis", "variación del diámetro (CV)"),
             _tile(f"{s['pal_alta']:.0f}%", "palidez central marcada",
                   f"índice > {PALIDEZ_MAX:g} (provisional) · mediana {s['pal_med']:.2f}, p90 {s['pal_p90']:.2f}"
                   + (" · resolución baja: poco fiable" if s["d_px"] < 40 else "")),
             _tile(f"{s['alarg']:.0f}%", "formas alargadas", "posibles eliptocitos (sin contar superpuestas)"),
             _tile(f"{s['centro']:.0f}%", "centro oscuro", "posibles dianocitos")]
    if um:
        tiles += [_tile(f"{s['micro']:.0f}%", "microcitos", f"< {MICRO_UM:g} µm"), _tile(f"{s['macro']:.0f}%", "macrocitos", f"> {MACRO_UM:g} µm")]
    sin_cal = "" if um else _aviso("Sin calibración: las medidas están en píxeles y no se clasifican micro/macrocitos.", "info")

    if r["probs"] is None:
        modelo = f'<div class="m-card"><div class="m-card-t">Modelo de machine learning</div>{_aviso(r["estado"], "alto")}</div>'
    else:
        variante = "con tamaño (calibrada)" if r["estado"] == "completo" else "sin tamaño (no calibrada)"
        filas = "".join(_barra(NOMBRES.get(o, o), p) for o, p in r["probs"].items())
        modelo = (f'<div class="m-card"><div class="m-card-t">Modelo de machine learning<span>AneRBC · {variante}</span></div>{filas}'
                  '<div class="m-nota m-pie-card">Probabilidad estimada por el modelo, no un diagnóstico.</div></div>')

    pocas = ""
    if len(df) < 200:
        pocas = _aviso(f"Solo {len(df)} células: el modelo se entrenó con ~1.000 células por muestra. "
                       "Sube más campos del mismo frotis (a 100x suelen hacer falta 10 o más).", "medio")
    columnas = 4 if len(tiles) % 4 == 0 else 3          # filas parejas: 6 = 3+3, 8 = 4+4
    return (f'<div class="m-res">{_interpretacion(r, s)}<div class="m-tiles" style="--cols:{columnas}">{"".join(tiles)}</div>{sin_cal}{modelo}{pocas}'
            f'{_aviso(AVISO_VALIDACION, "medio")}{LEYENDA}</div>')


def analizar_ui(archivos, um_por_px, metodo, progreso=gr.Progress()):
    r = procesar(archivos, um_por_px, metodo, progreso)
    if not r["ok"]:
        return [], _aviso(r["msg"], "medio"), pd.DataFrame()
    return r["imagenes"], _html_resultados(r), _tabla(r["df"])


def limpiar():
    return None, VACIO, [], pd.DataFrame()


CSS = """
:root{--m-tinta:#2b1a1e;--m-suave:#7a6165;--m-card:#fff;--m-borde:#f0d9dd;--m-pista:#f5e7e9;
      --m-ok:#2f9e77;--m-medio:#d99a1e;--m-alto:#c8243b;--m-fondo-tile:#fff6f6}
.dark{--m-tinta:#f6e9eb;--m-suave:#c4abb0;--m-card:#25171a;--m-borde:#4d2c33;--m-pista:#3a2227;--m-fondo-tile:#2c1a1e}
.gradio-container{max-width:1280px!important}
.m-hero{position:relative;overflow:hidden;border-radius:20px;padding:30px 34px;color:#fff;
  background:linear-gradient(115deg,#8a182b 0%,#5e0f20 55%,#3b0a15 100%);box-shadow:0 14px 34px -16px rgba(120,20,40,.7)}
.m-hero svg{position:absolute;right:0;top:0;height:100%;width:62%;pointer-events:none;
  -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 38%);mask-image:linear-gradient(90deg,transparent 0,#000 38%)}
.m-hero-txt{position:relative;max-width:620px}
.m-hero h1{margin:10px 0 8px;font-size:2.05rem;line-height:1.1;font-weight:750;letter-spacing:-.02em;color:#fff}
.m-hero p{margin:0;font-size:1rem;line-height:1.55;color:#fde8ea;opacity:.95}
.m-chips span{display:inline-block;margin:0 6px 4px 0;padding:3px 11px;border-radius:999px;font-size:.74rem;font-weight:600;
  letter-spacing:.04em;text-transform:uppercase;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);color:#fff}
@media(max-width:760px){.m-hero{padding:22px}.m-hero svg{opacity:.22;width:100%}.m-hero h1{font-size:1.55rem}}
.m-vacio{text-align:center;padding:46px 20px;color:var(--m-suave)}
.m-vacio svg{width:150px;margin-bottom:6px}.m-vacio h3{margin:6px 0;color:var(--m-tinta);font-size:1.2rem}
.m-vacio p{margin:0;line-height:1.6}
.m-tiles{display:grid;grid-template-columns:repeat(var(--cols,3),minmax(0,1fr));gap:12px;margin-bottom:14px}
@media(max-width:620px){.m-tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}
.m-tile{background:var(--m-fondo-tile);border:1px solid var(--m-borde);border-radius:14px;padding:14px 16px;border-top:3px solid #c8243b}
.m-val{font-size:1.7rem;font-weight:750;color:var(--m-tinta);line-height:1.1;letter-spacing:-.02em}.m-val small{font-size:.85rem;font-weight:600;color:var(--m-suave)}
.m-lab{margin-top:3px;font-size:.78rem;font-weight:650;text-transform:uppercase;letter-spacing:.05em;color:#c8243b}
.m-nota{margin-top:3px;font-size:.78rem;color:var(--m-suave);line-height:1.35}
.m-card{background:var(--m-card);border:1px solid var(--m-borde);border-radius:16px;padding:18px 20px;margin-bottom:14px}
.m-card-t{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:12px;
  font-size:1.02rem;font-weight:700;color:var(--m-tinta)}
.m-card-t span{font-size:.76rem;font-weight:600;color:var(--m-suave);background:var(--m-pista);padding:2px 10px;border-radius:999px}
.m-fila{display:grid;grid-template-columns:minmax(120px,210px) 1fr 52px;align-items:center;gap:12px;margin:9px 0}
.m-nom{font-size:.9rem;font-weight:600;color:var(--m-tinta)}
.m-pista{height:12px;background:var(--m-pista);border-radius:999px;overflow:hidden}
.m-barra{height:100%;border-radius:999px;transition:width .6s ease}
.m-barra.baja{background:linear-gradient(90deg,#2f9e77,#55c79d)}.m-barra.media{background:linear-gradient(90deg,#d99a1e,#f0bd4f)}
.m-barra.alta{background:linear-gradient(90deg,#a61a2f,#e53e52)}
.m-pct{text-align:right;font-weight:750;font-size:.95rem;color:var(--m-tinta)}
.m-pie-card{margin-top:12px}
.m-interp{border-left:5px solid #c8243b}
.m-card-t .m-badge{font-weight:700;letter-spacing:.02em;padding:3px 12px}
.m-card-t .m-b-ok{background:rgba(47,158,119,.18);color:#2f9e77}
.m-card-t .m-b-med{background:rgba(217,154,30,.2);color:#a8700a}
.dark .m-card-t .m-b-med{color:#f0bd4f}
.m-card-t .m-b-no{background:rgba(120,120,130,.2);color:var(--m-suave)}
.m-lead{margin:0 0 4px;font-size:1rem;line-height:1.55;color:var(--m-tinta)}
.m-sub{margin-top:12px;font-size:.74rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#c8243b}
.m-interp ul{margin:6px 0 0 18px;padding:0;font-size:.9rem;line-height:1.55;color:var(--m-tinta)}
.m-interp li{margin:4px 0}
.m-aviso{border-radius:12px;padding:11px 14px;margin:0 0 12px;font-size:.86rem;line-height:1.5;border:1px solid;color:var(--m-tinta)}
.m-aviso-info{background:rgba(70,120,200,.09);border-color:rgba(70,120,200,.3)}
.m-aviso-medio{background:rgba(217,154,30,.12);border-color:rgba(217,154,30,.4)}
.m-aviso-alto{background:rgba(200,36,59,.09);border-color:rgba(200,36,59,.35)}
.m-leyenda{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;font-size:.82rem;color:var(--m-suave);margin-top:4px}
.m-leyenda b{color:var(--m-tinta)}.m-leyenda span{display:inline-flex;align-items:center;gap:6px}
.m-leyenda i{display:inline-block;width:11px;height:11px;border-radius:50%}
.m-pie{margin-top:6px;text-align:center;font-size:.78rem;color:var(--m-suave);line-height:1.5}
@media(max-width:620px){.m-fila{grid-template-columns:1fr 44px}.m-fila .m-pista{grid-column:1 / span 2;order:3}}
"""

# Escala "sangre": del rosado de la tinción al granate del frotis
SANGRE = gr.themes.Color(c50="#fff1f2", c100="#ffe0e3", c200="#ffc7cd", c300="#ff9fa9", c400="#f86a79", c500="#e53e52",
                         c600="#c8243b", c700="#a61a2f", c800="#8a182b", c900="#751929", c950="#410811", name="sangre")
TEMA = gr.themes.Soft(primary_hue=SANGRE, secondary_hue=SANGRE, neutral_hue=gr.themes.colors.stone,
                      font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"]).set(
    body_background_fill="#fff8f8", body_background_fill_dark="#150b0d",
    button_primary_background_fill="*primary_600", button_primary_background_fill_hover="*primary_700",
    button_primary_background_fill_dark="*primary_600", button_primary_background_fill_hover_dark="*primary_500",
    button_primary_text_color="white")

with gr.Blocks(title="Morfología de eritrocitos · prototipo v1") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=340):
            archivos = gr.File(file_count="multiple", file_types=["image"], label="1 · Fotos del frotis (uno o varios campos del MISMO paciente)")
            um = gr.Number(value=0, label="2 · Calibración: µm por píxel (0 = sin calibrar)", minimum=0)
            metodo = gr.Radio([METODO_CELLPOSE, METODO_CLASICO], value=METODO_CELLPOSE, label="3 · Segmentación de células")
            with gr.Row():
                limpiar_btn = gr.Button("Limpiar", variant="secondary")
                analizar_btn = gr.Button("Analizar frotis", variant="primary", scale=2)
            with gr.Accordion("Cómo calibrar (µm por píxel)", open=False):
                gr.Markdown("Fotografía un **portaobjetos micrométrico** con el mismo microscopio, objetivo y zoom del celular. "
                            "Luego: *µm por píxel = distancia conocida (µm) ÷ píxeles que ocupa en la foto*. "
                            "Si cambia cualquier parte del montaje, hay que recalibrar. Sin calibración se miden píxeles.")
            with gr.Accordion("Cómo tomar una buena foto", open=False):
                gr.Markdown("- Zona de **monocapa**: células separadas, sin apilarse.\n- Objetivo **100x**, bien enfocada y con luz pareja.\n"
                            "- Varios campos del mismo frotis (a 100x entran ~10–20 células por foto).\n- Sin nombres de pacientes en el archivo.")
        with gr.Column(scale=7, min_width=400):
            with gr.Tabs():
                with gr.Tab("Resumen"):
                    resultado = gr.HTML(VACIO)
                with gr.Tab("Células detectadas"):
                    galeria = gr.Gallery(label="Contornos por categoría", columns=2, object_fit="contain", height="auto")
                with gr.Tab("Detalle por célula"):
                    tabla = gr.Dataframe(label="Una fila por eritrocito")

    ejemplos = [(RAIZ / "datos/AneRBC-I/Anemic/Original_images/001_a.png", "AneRBC · paciente con anemia"),
                (RAIZ / "datos/AneRBC-I/Healthy/Original_images/001_h.png", "AneRBC · paciente sano")]
    ejemplos = [[[str(p)], 0, METODO_CLASICO] for p, _ in ejemplos if p.exists()]
    if ejemplos:
        gr.Examples(examples=ejemplos, inputs=[archivos, um, metodo], label="Ejemplos (AneRBC: primero anémico, luego sano)")
    gr.HTML(PIE)

    analizar_btn.click(analizar_ui, [archivos, um, metodo], [galeria, resultado, tabla], api_name="analizar")
    limpiar_btn.click(limpiar, None, [archivos, resultado, galeria, tabla], api_name=False)

if __name__ == "__main__":
    demo.launch(theme=TEMA, css=CSS)
