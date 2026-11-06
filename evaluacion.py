# -*- coding: utf-8 -*-
"""
evaluacion.py

Evaluación FINAL de las 4 arquitecturas sobre el conjunto de TEST (50 sujetos),
siguiendo la sección 4.4 del anteproyecto:

  1. Solapamiento:  Dice (F1) e IoU (Jaccard)
  2. Diagnóstico:   Sensibilidad, Especificidad, Precisión (PPV)
  3. Clínico-geométricas: HD95 en mm y AVD en % (|Vpred - Vgt| / Vgt * 100)

Todo se calcula POR PACIENTE sobre el volumen 3D completo: los modelos 2D
predicen corte a corte y los cortes se apilan de nuevo en un volumen.
Como el preprocesamiento v2 remuestrea a 1 mm isotrópico, HD95 sale en mm
reales con spacing=(1, 1, 1).

Decisiones de protocolo (iguales para los 4 modelos; documentarlas en la memoria):
  - Umbral fijo 0.5 sobre la sigmoide, sin ajustarlo con el test.
  - Solo se predicen los cortes "con cerebro" (FLAIR distinto de cero, mismo criterio
    del loader pero SIN usar la máscara real); el resto se deja en 0. Esos cortes
    nunca se vieron en entrenamiento.
  - Modelos: los pesos con menor pérdida de VALIDACIÓN (best_<modelo>.pth).
  - El test se usa UNA sola vez. Si cambias algo del modelo tras ver estos números,
    el test deja de ser independiente.

Resultados en CSV, un renglón por (modelo, sujeto); se escribe tras cada paciente y
se reanuda solo si Colab se desconecta.

Las importaciones de torch/monai son perezosas para poder probar las métricas sin GPU.
"""

import os
import csv
import time
import shutil
import itertools
import numpy as np
from scipy import ndimage as ndi
from scipy import stats

MODELOS = ["UNet_2D", "AttentionUNet_2D", "SegNet_2D", "VNet_2D"]
CAMPOS_METRICAS = ["dice", "jaccard", "sensibilidad", "especificidad", "precision",
                   "hd95_mm", "avd_pct"]
CAMPOS_CSV = (["modelo", "sujeto"] + CAMPOS_METRICAS +
              ["vol_gt_ml", "vol_pred_ml", "pred_vacia", "t_inferencia_s"])

SUFIJOS = {"_FLAIR_proc.npy": np.float16, "_ADC_proc.npy": np.float16,
           "_dwi_proc.npy": np.float16, "_msk_proc.npy": np.uint8}


# ==============================================================================
# MÉTRICAS (numpy/scipy puro)
# ==============================================================================
def _superficie(m):
    """Vóxeles de borde de una máscara 3D (conectividad 6). Fuera del volumen = fondo."""
    est = ndi.generate_binary_structure(3, 1)
    return m & ~ndi.binary_erosion(m, structure=est, border_value=0)


def _caja_union(a, b, margen=1):
    """Slices de la caja mínima que contiene a ambas máscaras (recorta el coste del EDT)."""
    idx = np.argwhere(a | b)
    lo = np.maximum(idx.min(0) - margen, 0)
    hi = np.minimum(idx.max(0) + 1 + margen, a.shape)
    return tuple(slice(l, h) for l, h in zip(lo, hi))


def hd95_mm(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """
    HD95 simétrico en mm: percentil 95 de la UNIÓN de las distancias superficiales
    pred->gt y gt->pred (una de las dos variantes habituales; la otra toma el máximo de
    los dos percentiles dirigidos). NaN si alguna de las máscaras está vacía.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    caja = _caja_union(pred, gt)
    p, g = pred[caja], gt[caja]
    sp, sg = _superficie(p), _superficie(g)
    dist_a_g = ndi.distance_transform_edt(~sg, sampling=spacing)   # distancia a la superficie gt
    dist_a_p = ndi.distance_transform_edt(~sp, sampling=spacing)   # distancia a la superficie pred
    d = np.concatenate([dist_a_g[sp], dist_a_p[sg]])
    return float(np.percentile(d, 95))


def metricas_volumen(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """Métricas de la sección 4.4 para UN paciente (volúmenes 3D booleanos)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())

    def div(a, b):
        return a / b if b > 0 else float("nan")

    vg, vp = tp + fn, tp + fp
    vox_ml = float(np.prod(spacing)) / 1000.0            # 1 mm^3 = 0.001 mL
    return {
        "dice": div(2 * tp, 2 * tp + fp + fn),
        "jaccard": div(tp, tp + fp + fn),
        "sensibilidad": div(tp, tp + fn),
        "especificidad": div(tn, tn + fp),
        "precision": div(tp, tp + fp),                    # NaN si no se predijo nada
        "hd95_mm": hd95_mm(pred, gt, spacing),            # NaN si pred o gt vacías
        "avd_pct": div(abs(vp - vg), vg) * 100.0,         # % respecto al volumen real
        "vol_gt_ml": vg * vox_ml,
        "vol_pred_ml": vp * vox_ml,
        "pred_vacia": int(vp == 0),
    }


# ==============================================================================
# INFERENCIA
# ==============================================================================
def cortes_con_cerebro(flair):
    """Boolean por corte: hay señal en FLAIR (mismo criterio del loader, sin usar la máscara)."""
    return (flair.reshape(flair.shape[0], -1) != 0).any(axis=1)


def construir_modelo(nombre):
    """Debe reproducir EXACTAMENTE la arquitectura usada al entrenar (load_state_dict es estricto)."""
    if nombre == "UNet_2D":
        from monai.networks.nets import UNet
        return UNet(spatial_dims=2, in_channels=3, out_channels=1,
                    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    if nombre == "AttentionUNet_2D":
        from monai.networks.nets import AttentionUnet
        return AttentionUnet(spatial_dims=2, in_channels=3, out_channels=1,
                             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    if nombre == "SegNet_2D":
        from segnet import SegNet
        return SegNet(in_channels=3, out_channels=1, channels=(16, 32, 64, 128, 256))
    if nombre == "VNet_2D":
        from vnet2d import VNet2D
        return VNet2D(in_channels=3, out_channels=1)
    raise ValueError(f"Modelo desconocido: {nombre}")


def cargar_modelo(nombre, dir_resultados, device):
    """Devuelve (modelo en eval, nº de parámetros)."""
    import torch
    ruta = os.path.join(dir_resultados, nombre, f"best_{nombre}.pth")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe {ruta}")
    modelo = construir_modelo(nombre)
    modelo.load_state_dict(torch.load(ruta, map_location="cpu"))
    modelo = modelo.to(device).eval()
    n_params = sum(p.numel() for p in modelo.parameters())
    return modelo, n_params


def predecir_volumen(modelo, dwi, adc, flair, device, umbral=0.5, batch=32):
    """Predice corte a corte (solo cortes con cerebro) y devuelve el volumen booleano."""
    import torch
    pred = np.zeros(flair.shape, dtype=bool)
    idx = np.flatnonzero(cortes_con_cerebro(flair))
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            z = idx[i:i + batch]
            x = np.stack([dwi[z], adc[z], flair[z]], axis=1).astype(np.float32)   # orden DWI, ADC, FLAIR
            logits = modelo(torch.from_numpy(x).to(device))
            pred[z] = (torch.sigmoid(logits.float())[:, 0] > umbral).cpu().numpy()
    return pred


# ==============================================================================
# DATOS DE TEST (copia compacta Drive -> disco local, idempotente)
# ==============================================================================
def id_base(s):
    return s.split("_FLAIR")[0]


def _copiar_compacto(src, dst, dtype, intentos=3):
    if os.path.exists(dst):
        return True
    tmp = dst + ".tmp.npy"
    for k in range(intentos):
        try:
            arr = np.load(src)
            arr = (arr > 0.5) if dtype == np.uint8 else arr
            np.save(tmp, arr.astype(dtype))
            os.replace(tmp, dst)
            return True
        except (OSError, ValueError) as e:
            print(f"  intento {k + 1}/{intentos} falló en {os.path.basename(src)}: {e}")
            time.sleep(5)
    return False


def preparar_test(ids_test, dir_drive, dir_local):
    """Copia los sujetos de TEST al disco local. Falla con mensaje claro si Drive se cae."""
    os.makedirs(dir_local, exist_ok=True)
    fallidos = []
    for i, sid in enumerate(ids_test, 1):
        for suf, dt in SUFIJOS.items():
            if not _copiar_compacto(os.path.join(dir_drive, sid + suf),
                                    os.path.join(dir_local, sid + suf), dt):
                fallidos.append(sid + suf)
        if i % 10 == 0 or i == len(ids_test):
            print(f"  {i}/{len(ids_test)} sujetos de test | disco libre "
                  f"{shutil.disk_usage(dir_local).free / 1e9:.1f} GB")
        if len(fallidos) >= 8:
            break
    if fallidos:
        raise RuntimeError(f"No se pudieron copiar {len(fallidos)} archivos (ej.: {fallidos[:3]}). "
                           "Remonta Drive con force_remount=True y vuelve a ejecutar.")


# ==============================================================================
# BUCLE PRINCIPAL (con reanudación)
# ==============================================================================
def _leer_hechos(ruta_csv):
    if not os.path.exists(ruta_csv):
        return set()
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        return {(r["modelo"], r["sujeto"]) for r in csv.DictReader(f)}


def _anexar_filas(ruta_csv, filas):
    nuevo = not os.path.exists(ruta_csv)
    with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        if nuevo:
            w.writeheader()
        w.writerows(filas)


def evaluar(ids_test, dir_npy, dir_resultados, ruta_csv, modelos=MODELOS, device="cuda",
            umbral=0.5, spacing=(1.0, 1.0, 1.0)):
    """
    Evalúa cada modelo en cada paciente de test. Un paciente se carga UNA vez y se
    pasa por los 4 modelos. Devuelve {modelo: n_parametros}.
    """
    cuda = str(device).startswith("cuda")
    if cuda:
        import torch
    hechos = _leer_hechos(ruta_csv)
    cargados = {m: cargar_modelo(m, dir_resultados, device) for m in modelos}
    params = {m: n for m, (_, n) in cargados.items()}
    print("Parámetros:", {m: f"{n:,}" for m, n in params.items()})

    primero = True
    for k, sid in enumerate(ids_test, 1):
        pendientes = [m for m in modelos if (m, sid) not in hechos]
        if not pendientes:
            continue
        cargar = lambda suf: np.load(os.path.join(dir_npy, sid + suf))
        dwi, adc, flair, gt = cargar("_dwi_proc.npy"), cargar("_ADC_proc.npy"), \
            cargar("_FLAIR_proc.npy"), cargar("_msk_proc.npy") > 0
        if not (dwi.shape == adc.shape == flair.shape == gt.shape):
            raise RuntimeError(f"{sid}: formas distintas entre modalidades")

        filas = []
        for m in pendientes:
            modelo = cargados[m][0]
            if primero and cuda:                                   # calentamiento de CUDA
                predecir_volumen(modelo, dwi[:4], adc[:4], flair[:4], device)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            pred = predecir_volumen(modelo, dwi, adc, flair, device, umbral)
            if cuda:
                torch.cuda.synchronize()
            t_inf = time.time() - t0
            fila = {"modelo": m, "sujeto": sid, **metricas_volumen(pred, gt, spacing),
                    "t_inferencia_s": t_inf}
            filas.append(fila)
        primero = False
        _anexar_filas(ruta_csv, filas)
        print(f"[{k}/{len(ids_test)}] {sid} | " +
              " | ".join(f"{f['modelo']}: Dice {f['dice']:.3f}" for f in filas))
    return params


# ==============================================================================
# RESUMEN Y ESTADÍSTICA
# ==============================================================================
def holm(p):
    """Corrección de Holm-Bonferroni para comparaciones múltiples."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    ajustado = np.empty(m)
    maximo = 0.0
    for rango, i in enumerate(np.argsort(p)):
        maximo = max(maximo, (m - rango) * p[i])
        ajustado[i] = min(1.0, maximo)
    return ajustado


def resumen(ruta_csv, params=None):
    """Tabla por modelo: media ± DE, mediana [Q1-Q3], nº válidos y nº de predicciones vacías."""
    import pandas as pd
    df = pd.read_csv(ruta_csv)
    filas = []
    for m, g in df.groupby("modelo", sort=False):
        fila = {"modelo": m, "n_pacientes": len(g), "pred_vacia": int(g["pred_vacia"].sum()),
                "t_inferencia_s": g["t_inferencia_s"].mean()}
        if params and m in params:
            fila["parametros"] = params[m]
        for c in CAMPOS_METRICAS:
            x = g[c].dropna()
            fila[f"{c}_media±DE"] = f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"
            fila[f"{c}_mediana[Q1-Q3]"] = f"{x.median():.3f} [{x.quantile(.25):.3f}-{x.quantile(.75):.3f}]"
            fila[f"{c}_n"] = len(x)
        filas.append(fila)
    return pd.DataFrame(filas)


def comparar_pares(ruta_csv, metrica, n_boot=10000, seed=42):
    """
    Comparación pareada por paciente entre todos los pares de modelos:
    Wilcoxon (bilateral), diferencia de medianas, IC95% bootstrap de la diferencia
    de medias y p ajustado por Holm. Solo usa pacientes con valor válido en AMBOS modelos.
    """
    import pandas as pd
    df = pd.read_csv(ruta_csv)
    ancho = df.pivot(index="sujeto", columns="modelo", values=metrica)
    rng = np.random.default_rng(seed)
    filas = []
    for a, b in itertools.combinations(ancho.columns, 2):
        par = ancho[[a, b]].dropna()
        d = (par[a] - par[b]).to_numpy()
        n = len(d)
        if n < 5 or np.allclose(d, 0):
            p = float("nan")
        else:
            p = float(stats.wilcoxon(par[a], par[b]).pvalue)
        if n > 0:
            medias = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
            lo, hi = np.percentile(medias, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        filas.append({"metrica": metrica, "A": a, "B": b, "n_pares": n,
                      "mediana_A-B": float(np.median(d)) if n else float("nan"),
                      "media_A-B": float(d.mean()) if n else float("nan"),
                      "IC95_inf": lo, "IC95_sup": hi, "p_wilcoxon": p})
    out = pd.DataFrame(filas)
    validos = out["p_wilcoxon"].notna()
    out["p_holm"] = np.nan
    out.loc[validos, "p_holm"] = holm(out.loc[validos, "p_wilcoxon"])
    return out
