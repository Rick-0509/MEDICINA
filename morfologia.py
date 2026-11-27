"""Segmentación de eritrocitos y medición por célula. Compartido por la app y el entrenamiento."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, feature, filters, measure, morphology, segmentation, util


def _a_rgb(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return util.img_as_float(img[..., :3])


def segmentar(img):
    """Devuelve (etiquetas, mascara_teñida, radio_típico_px).

    - Separa células del fondo con Otsu sobre el canal verde (el fondo es claro).
    - Estima el radio típico usando solo componentes compactos (células aisladas),
      para que grumos y fragmentos no distorsionen la escala.
    - Separa células que se tocan con watershed sobre la transformada de distancia.
    """
    rgb = _a_rgb(img)
    verde = filters.gaussian(rgb[..., 1], sigma=1)
    # Otsu sin los píxeles muy oscuros (núcleos de leucocitos), que sesgan el umbral
    claros = verde[verde > np.percentile(verde, 2)]
    teñido = verde < filters.threshold_otsu(claros)
    teñido = morphology.remove_small_objects(teñido, max_size=12)
    lleno = ndi.binary_fill_holes(teñido)

    comp = measure.label(lleno)
    props = measure.regionprops(comp)
    if not props:
        return comp, teñido, 0.0
    compactos = [p.area for p in props if p.solidity > 0.9 and p.eccentricity < 0.8 and p.area > 30]
    area_tipica = np.median(compactos) if len(compactos) >= 5 else np.median([p.area for p in props])
    radio = float(np.sqrt(area_tipica / np.pi))

    dist = ndi.distance_transform_edt(lleno)
    dist_suave = filters.gaussian(dist, sigma=max(1.0, radio * 0.15))
    picos = feature.peak_local_max(dist_suave, min_distance=max(3, int(radio * 0.7)),
                                   threshold_abs=radio * 0.35, labels=comp, exclude_border=False)
    marcadores = np.zeros(comp.shape, dtype=np.int32)
    marcadores[tuple(picos.T)] = np.arange(1, len(picos) + 1)
    etiquetas = segmentation.watershed(-dist_suave, marcadores, mask=lleno)
    etiquetas = segmentation.clear_border(etiquetas)
    return etiquetas, teñido, radio


def medir(img, etiquetas, teñido, radio):
    """Tabla con una fila por eritrocito válido (excluye leucocitos, plaquetas y grumos)."""
    rgb = _a_rgb(img)
    gris = color.rgb2gray(rgb)
    hsv = color.rgb2hsv(rgb)
    fondo = gris[~ndi.binary_dilation(teñido, iterations=3)]
    brillo_fondo = float(np.median(fondo)) if fondo.size else 1.0
    area_tipica = np.pi * radio ** 2

    props = measure.regionprops(etiquetas, intensity_image=gris)
    de_tamano = [p for p in props if 0.3 * area_tipica <= p.area <= 2.2 * area_tipica]
    excluidas = len(props) - len(de_tamano)

    # Leucocitos/artefactos: núcleo mucho más oscuro que el fondo. El umbral absoluto se calibró con
    # AneRBC/BCCD; con una tinción densa (p. ej. violeta intenso) descartaría células normales. Si lo
    # hace con >25% de las células, se cambia a un criterio relativo a las demás células de la foto.
    oscuras = np.array([p.intensity_min < 0.45 * brillo_fondo for p in de_tamano], dtype=bool)
    if oscuras.size and oscuras.mean() > 0.25:
        medias = np.array([p.intensity_mean for p in de_tamano])
        oscuras = medias < 0.6 * np.median(medias)

    filas = []
    for p, oscura in zip(de_tamano, oscuras):
        if oscura:
            excluidas += 1
            continue
        r0, c0, r1, c1 = p.bbox
        celula = p.image
        sat = hsv[r0:r1, c0:c1, 1][celula]
        # perfil radial (0 = centro, 1 = borde)
        cy, cx = p.centroid_local
        yy, xx = np.nonzero(celula)
        rr = np.hypot(yy - cy, xx - cx) / max(p.equivalent_diameter_area / 2, 1)
        g = gris[r0:r1, c0:c1][celula]
        # palidez central: píxeles centrales más claros que el punto medio entre el anillo teñido
        # de la propia célula y el fondo (no depende del umbral global ni del contorno exacto)
        anillo = g[(rr >= 0.55) & (rr < 0.95)]
        borde = float(np.median(anillo)) if anillo.size else float(g.mean())
        umbral_palidez = borde + 0.5 * (brillo_fondo - borde)
        palido = (rr < 0.8) & (g > umbral_palidez)
        centro = g[rr < 0.25].mean() if (rr < 0.25).any() else np.nan
        medio = g[(rr >= 0.25) & (rr < 0.6)].mean() if ((rr >= 0.25) & (rr < 0.6)).any() else np.nan
        filas.append({
            "id": p.label,
            "diametro_px": p.equivalent_diameter_area,
            "elongacion": p.axis_minor_length / p.axis_major_length if p.axis_major_length else 1.0,
            "solidez": p.solidity,
            "palidez": float(np.sqrt(palido.sum() / p.area)),
            "densidad": 1 - p.intensity_mean / brillo_fondo,     # qué tan teñida (hemoglobina)
            "saturacion": float(sat.mean()),
            "centro_vs_medio": (medio - centro) / brillo_fondo if np.isfinite(centro + medio) else 0.0,
        })
    return pd.DataFrame(filas), excluidas


def mascara_teñida(img):
    rgb = _a_rgb(img)
    verde = filters.gaussian(rgb[..., 1], sigma=1)
    claros = verde[verde > np.percentile(verde, 2)]
    return morphology.remove_small_objects(verde < filters.threshold_otsu(claros), max_size=12)


def segmentar_cellpose(img, modelo):
    """Segmentación con Cellpose (red pre-entrenada). Mejor con células superpuestas o 100x."""
    rgb = np.asarray(img)[..., :3]
    etiquetas, _, _, _ = modelo.eval(rgb, diameter=None, channels=[0, 0])
    etiquetas = segmentation.clear_border(etiquetas.astype(np.int32))
    areas = np.bincount(etiquetas.ravel())[1:]
    areas = areas[areas > 0]
    radio = float(np.sqrt(np.median(areas) / np.pi)) if areas.size else 0.0
    return etiquetas, mascara_teñida(img), radio


def analizar_celulas(img, segmentador=None):
    """segmentador=None usa el método clásico; si se pasa un modelo Cellpose, usa Cellpose."""
    if segmentador is None:
        etiquetas, teñido, radio = segmentar(img)
    else:
        etiquetas, teñido, radio = segmentar_cellpose(img, segmentador)
    df, excluidas = medir(img, etiquetas, teñido, radio)
    return etiquetas, df, excluidas, radio


def resumen_imagen(df, um_por_px):
    """Características agregadas de una imagen (entrada del modelo de ML)."""
    if len(df) < 20:
        return None
    d = df["diametro_px"] * um_por_px
    q = lambda s, x: float(np.quantile(s, x))
    return {
        "n_celulas": len(df),
        "diam_mediana": float(d.median()), "diam_p10": q(d, .1), "diam_p90": q(d, .9),
        "diam_cv": float(d.std() / d.mean()),
        "frac_micro": float((d < 6.0).mean()), "frac_macro": float((d > 9.0).mean()),
        "palidez_mediana": float(df["palidez"].median()), "palidez_p90": q(df["palidez"], .9),
        "frac_palidez_alta": float((df["palidez"] > 0.5).mean()),
        "elong_mediana": float(df["elongacion"].median()),
        "frac_alargadas": float((df["elongacion"] < 0.7).mean()),
        "frac_irregulares": float((df["solidez"] < 0.9).mean()),
        "densidad_mediana": float(df["densidad"].median()),
        "saturacion_mediana": float(df["saturacion"].median()),
        # centro más oscuro que el anillo intermedio (patrón de dianocito)
        "frac_centro_oscuro": float((df["centro_vs_medio"] > 0.03).mean()),
    }
