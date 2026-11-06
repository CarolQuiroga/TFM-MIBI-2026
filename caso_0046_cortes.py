# -*- coding: utf-8 -*-
"""
caso_0046_cortes.py
Análisis visual y cuantitativo del caso con Dice = 0 en los 4 modelos (sub-strokecase0046).

Hace dos cosas:
  1. Figura de cortes axiales: DWI, ADC y FLAIR con el contorno de la lesión real, y la
     predicción de cada arquitectura superpuesta sobre DWI. Se muestran el corte con más
     lesión real y los cortes con el falso positivo más grande de cada modelo.
  2. Tabla de los falsos positivos (componentes conexas): volumen, corte, distancia al
     borde del cerebro y distancia a la lesión real. Sirve para afirmar (o descartar) con
     números que el falso positivo es "periférico".

Uso en Colab (después de haber ejecutado Evaluacion.ipynb, que deja el test en /content/npy_test):
    import sys; sys.path.append('/content/drive/MyDrive/TFM')
    !python /content/drive/MyDrive/TFM/caso_0046_cortes.py

Requisitos: evaluacion.py en /content/drive/MyDrive/TFM (usa cargar_modelo y
predecir_volumen, así la predicción es idéntica a la de la evaluación: umbral 0,5,
solo cortes con cerebro).

ORIENTACIÓN: se asume que el eje 0 de los .npy es el axial (igual que en el
entrenamiento, que predice corte a corte sobre el eje 0). Si la imagen sale girada o
volteada, ajusta ROTAR / VOLTEAR más abajo: solo cambian la visualización, no los cálculos.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Configuración ----------------
SUJETO = "sub-strokecase0046"
DIR_TFM = "/content/drive/MyDrive/TFM"
DIR_NPY = "/content/npy_test"                       # copia local creada por Evaluacion.ipynb
DIR_RES = os.path.join(DIR_TFM, "resultados")       # contiene <modelo>/best_<modelo>.pth
DIR_FIG = os.path.join(DIR_RES, "figuras")
DEVICE = "cuda"
ROTAR = 0          # con ROTAR = 1 la parte anterior salía a la izquierda; con 0 debería quedar arriba (comprobar)
VOLTEAR = False    # True para invertir izquierda-derecha en la figura
MAX_CORTES = 4

MODELOS = ["UNet_2D", "AttentionUNet_2D", "SegNet_2D", "VNet_2D"]
ETIQUETAS = {"UNet_2D": "U-Net", "AttentionUNet_2D": "Attention U-Net",
             "SegNet_2D": "SegNet", "VNet_2D": "V-Net 2D"}
COLORES = {"UNet_2D": "#2a78d6", "AttentionUNet_2D": "#eb6834",
           "SegNet_2D": "#1baf7a", "VNet_2D": "#4a3aa7"}
COLOR_GT = "#ffd400"     # contorno de la lesión real (amarillo, visible sobre gris)
TINTA, TINTA_2 = "#0b0b0b", "#52514e"


# ---------------- Análisis de falsos positivos ----------------
def analizar_fp(preds, gt, flair, dwi, adc, spacing=1.0):
    """
    Componentes conexas de FP por modelo, con su posición respecto al cerebro y a la lesión.
    Vectorizado (una sola pasada por modelo): rápido aunque haya miles de componentes.
    """
    cerebro = ndi.binary_fill_holes(flair != 0)
    dist_borde = ndi.distance_transform_edt(cerebro) * spacing        # mm hasta el borde del cerebro
    dist_gt = ndi.distance_transform_edt(~gt) * spacing if gt.any() else None
    # Las imágenes están normalizadas con z-score (pueden ser negativas), así que la
    # intensidad se expresa como PERCENTIL dentro del cerebro (0 = mínima, 100 = máxima).
    dwi_orden = np.sort(dwi[cerebro].ravel())
    adc_orden = np.sort(adc[cerebro].ravel())
    pct = lambda orden, v: 100.0 * np.searchsorted(orden, v) / orden.size
    conect = ndi.generate_binary_structure(3, 3)
    filas = []
    for m, p in preds.items():
        fp = p & ~gt
        lab, n = ndi.label(fp, structure=conect)
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        vol = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        centros = ndi.center_of_mass(fp, lab, idx)
        cajas = ndi.find_objects(lab)
        med_borde = ndi.median(dist_borde, lab, idx)
        min_gt = ndi.minimum(dist_gt, lab, idx) if dist_gt is not None else [np.nan] * n
        dwi_med = ndi.mean(dwi, lab, idx)
        adc_med = ndi.mean(adc, lab, idx)
        for k in range(n):
            filas.append({
                "modelo": ETIQUETAS[m], "componente": k + 1,
                "volumen_mL": vol[k] * spacing ** 3 / 1000,
                "corte_central": int(round(centros[k][0])),
                "cortes": f"{cajas[k][0].start}–{cajas[k][0].stop - 1}",
                "dist_borde_cerebro_mm (mediana)": float(med_borde[k]),
                "dist_lesion_real_mm (mínima)": float(min_gt[k]),
                "DWI_percentil": float(pct(dwi_orden, dwi_med[k])),
                "ADC_percentil": float(pct(adc_orden, adc_med[k])),
            })
    # Fila de referencia: la lesión real, para comparar cada mancha con un infarto anotado
    if gt.any():
        z = np.argwhere(gt)[:, 0]
        filas.insert(0, {
            "modelo": "LESIÓN REAL (referencia)", "componente": 0,
            "volumen_mL": gt.sum() * spacing ** 3 / 1000,
            "corte_central": int(round(z.mean())), "cortes": f"{z.min()}–{z.max()}",
            "dist_borde_cerebro_mm (mediana)": float(np.median(dist_borde[gt])),
            "dist_lesion_real_mm (mínima)": 0.0,
            "DWI_percentil": float(pct(dwi_orden, dwi[gt].mean())),
            "ADC_percentil": float(pct(adc_orden, adc[gt].mean())),
        })
    return pd.DataFrame(filas)


def resumen_fp(tabla):
    """Resumen por modelo: cuánto FP total, en cuántas manchas y qué parte está cerca del borde."""
    t = tabla[tabla.componente > 0].copy()          # sin la fila de la lesión real
    if t.empty:
        return t
    t["cerca_borde"] = t["dist_borde_cerebro_mm (mediana)"] <= 10
    g = t.groupby("modelo", sort=False)
    return pd.DataFrame({
        "FP_total_mL": g.volumen_mL.sum(),
        "n_componentes": g.size(),
        "FP_mayor_mL": g.volumen_mL.max(),
        "%FP_a_<=10mm_del_borde": 100 * t.volumen_mL.where(t.cerca_borde, 0).groupby(t.modelo, sort=False).sum()
                                  / g.volumen_mL.sum(),
    }).round(2)


def elegir_cortes(preds, gt, max_cortes=MAX_CORTES):
    """Corte con más lesión real + corte con más FP de cada modelo (sin repetir)."""
    cortes = []
    if gt.any():
        cortes.append(int(gt.sum(axis=(1, 2)).argmax()))
    for m in MODELOS:
        fp = preds[m] & ~gt
        if fp.any():
            z = int(fp.sum(axis=(1, 2)).argmax())
            if all(abs(z - c) > 1 for c in cortes):
                cortes.append(z)
    return sorted(cortes[:max_cortes])


# ---------------- Figura ----------------
def mostrar(img):
    img = np.rot90(img, ROTAR)
    return np.fliplr(img) if VOLTEAR else img


def ventana(vol):
    v = vol[vol != 0]
    lo, hi = np.percentile(v, [1, 99.5]) if v.size else (0, 1)
    return lo, hi


def figura(dwi, adc, flair, gt, preds, cortes, ruta_sin_ext):
    columnas = [("DWI", dwi), ("ADC", adc), ("FLAIR", flair)] + [(ETIQUETAS[m], m) for m in MODELOS]
    vent = {"DWI": ventana(dwi), "ADC": ventana(adc), "FLAIR": ventana(flair)}
    fig, axs = plt.subplots(len(cortes), len(columnas),
                            figsize=(2.1 * len(columnas), 2.2 * len(cortes) + 0.4),
                            squeeze=False)
    for i, z in enumerate(cortes):
        g = mostrar(gt[z])
        for j, (titulo, fuente) in enumerate(columnas):
            ax = axs[i, j]
            if isinstance(fuente, np.ndarray):                           # modalidades
                lo, hi = vent[titulo]
                ax.imshow(mostrar(fuente[z]), cmap="gray", vmin=lo, vmax=hi)
            else:                                                        # predicción sobre DWI
                lo, hi = vent["DWI"]
                ax.imshow(mostrar(dwi[z]), cmap="gray", vmin=lo, vmax=hi)
                p = mostrar(preds[fuente][z])
                capa = np.zeros(p.shape + (4,))
                capa[p] = matplotlib.colors.to_rgba(COLORES[fuente], 0.65)
                ax.imshow(capa)
                if p.any():
                    ax.contour(p, levels=[0.5], colors=[COLORES[fuente]], linewidths=0.8)
            if g.any():
                ax.contour(g, levels=[0.5], colors=[COLOR_GT], linewidths=1.1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(titulo, fontsize=9.5, fontweight="bold", color=TINTA)
        axs[i, 0].set_ylabel(f"corte {z}", fontsize=9, color=TINTA_2)
    fig.text(0.01, -0.01,
             f"{SUJETO}. Contorno amarillo: lesión real (referencia). Relleno de color: predicción "
             "de cada modelo (umbral 0,5) sobre DWI. Filas: corte con más lesión real y cortes "
             "con el mayor falso positivo de cada modelo.",
             fontsize=7.5, color=TINTA_2, wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{ruta_sin_ext}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- Programa principal ----------------
def main():
    sys.path.append(DIR_TFM)
    import evaluacion as ev
    os.makedirs(DIR_FIG, exist_ok=True)

    cargar = lambda suf: np.load(os.path.join(DIR_NPY, SUJETO + suf)).astype(np.float32)
    dwi, adc, flair = cargar("_dwi_proc.npy"), cargar("_ADC_proc.npy"), cargar("_FLAIR_proc.npy")
    gt = cargar("_msk_proc.npy") > 0
    print(f"{SUJETO}: forma {gt.shape}, lesión real {gt.sum() / 1000:.3f} mL en los cortes "
          f"{np.flatnonzero(gt.any(axis=(1, 2))).tolist()}")

    # Con PyTorch 2.11.0+cu130 (Colab, nov. 2026) SegNet da en GPU un resultado distinto al
    # de la evaluación (hecha con +cu128); en CPU lo reproduce exactamente. Por eso SegNet
    # se ejecuta en CPU. Cada predicción se contrasta con evaluacion_test.csv.
    csv_eval = os.path.join(DIR_RES, "evaluacion_test.csv")
    ref = pd.read_csv(csv_eval) if os.path.exists(csv_eval) else None
    preds = {}
    for m in MODELOS:
        dev = "cpu" if m == "SegNet_2D" else DEVICE
        modelo, _ = ev.cargar_modelo(m, DIR_RES, dev)
        preds[m] = ev.predecir_volumen(modelo, dwi, adc, flair, dev)
        vol = preds[m].sum() / 1000
        control = ""
        if ref is not None:
            fila = ref[(ref.modelo == m) & (ref.sujeto == SUJETO)]
            if len(fila):
                v_ref = float(fila.vol_pred_ml.iloc[0])
                ok = abs(vol - v_ref) < 0.01
                control = f" | CSV {v_ref:.3f} mL -> {'OK' if ok else 'NO COINCIDE: revisar'}"
        print(f"  {ETIQUETAS[m]} ({dev}): {vol:.3f} mL predichos, "
              f"solapamiento con la lesión real {np.logical_and(preds[m], gt).sum()} vóxeles{control}")

    tabla = analizar_fp(preds, gt, flair, dwi, adc)
    if len(tabla):
        ref_fila = tabla[tabla.componente == 0]
        tabla = pd.concat([ref_fila, tabla[tabla.componente > 0].sort_values(
            ["modelo", "volumen_mL"], ascending=[True, False])])
        tabla.to_csv(os.path.join(DIR_FIG, f"fp_{SUJETO}.csv"), index=False)
        pd.set_option("display.width", 200)
        print("\nResumen de falsos positivos por modelo:")
        print(resumen_fp(tabla).to_string())
        print("\nLesión real y las 5 manchas de FP más grandes de cada modelo (tabla completa en el CSV).")
        print("Percentil de intensidad media dentro del cerebro (0-100). Infarto agudo esperable: "
              "DWI alto (p. ej. >90) y ADC bajo (p. ej. <20). Artefacto o T2 shine-through: ADC no bajo.")
        print(tabla.groupby("modelo", sort=False).head(5).round(2).to_string(index=False))

    cortes = elegir_cortes(preds, gt)
    ruta = os.path.join(DIR_FIG, f"fig_cortes_{SUJETO}")
    figura(dwi, adc, flair, gt, preds, cortes, ruta)
    print(f"\nFigura guardada en {ruta}.png / .pdf (cortes {cortes})")


if __name__ == "__main__":
    main()
