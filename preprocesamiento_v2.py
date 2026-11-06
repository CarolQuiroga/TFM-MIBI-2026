# =============================================================================
# preprocesamiento_v2.py  ->  PARA Preprocesamiento.ipynb
#
# Problema que corrige: cada modalidad se remuestreaba a 1 mm sobre SU PROPIA
# rejilla, así que FLAIR y DWI/ADC/máscara terminaban con distinto número de
# cortes (ej. FLAIR 250 vs DWI 144) y el corte k no correspondía a la misma
# posición anatómica en todos los canales.
# Solución: definir UNA rejilla isotrópica a partir del DWI (la máscara de ISLES
# está en ese espacio) y remuestrear las 4 imágenes sobre esa misma rejilla.
#
# Orden de uso:
#   1) CELDA A: reemplaza la celda donde está definida preprocesamiento_caso()
#      (deja intactas cargar_y_estandarizar_orientacion, z_score_normalize y
#       crop_or_pad_volume).
#   2) CELDA B: ejecútala con 2-3 sujetos (p. ej. 0062, 0093 y 0001) y MIRA la figura.
#   3) Solo si la figura se ve alineada: pega la CELDA C y llama a procesar_dataset()
#      (unos 45-60 min; se puede relanzar y continúa donde quedó).
# =============================================================================

# ------------------------------- CELDA A -------------------------------------
import os
import numpy as np
import SimpleITK as sitk


def grid_isotropico_desde(ref, new_spacing=(1.0, 1.0, 1.0)):
    """Rejilla isotrópica que cubre la imagen de referencia (mismo origen y orientación)."""
    sp, sz = ref.GetSpacing(), ref.GetSize()
    size = [int(round(sz[i] * sp[i] / new_spacing[i])) for i in range(3)]
    return size, ref.GetOrigin(), tuple(new_spacing), ref.GetDirection()


def remuestrear_a_grid(img, grid, interpolator):
    size, origin, spacing, direction = grid
    f = sitk.ResampleImageFilter()
    f.SetSize(size)
    f.SetOutputSpacing(spacing)
    f.SetOutputOrigin(origin)
    f.SetOutputDirection(direction)
    f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(0)
    return f.Execute(img)      # transformación identidad: alinea por coordenadas físicas


def alinear_caso(flair_path, adc_path, dwi_path, mask_path):
    """Devuelve (flair, adc, dwi, mask) como imágenes SimpleITK sobre la MISMA rejilla de 1 mm."""
    dwi   = cargar_y_estandarizar_orientacion(dwi_path)
    adc   = cargar_y_estandarizar_orientacion(adc_path)
    flair = cargar_y_estandarizar_orientacion(flair_path)
    mask  = cargar_y_estandarizar_orientacion(mask_path)

    # La máscara y el ADC deben estar ya en la rejilla del DWI; si no, fallar con claridad.
    for nombre, im in (("ADC", adc), ("MSK", mask)):
        if im.GetSize() != dwi.GetSize():
            raise ValueError(f"{nombre} {im.GetSize()} no coincide con DWI {dwi.GetSize()}")

    grid = grid_isotropico_desde(dwi)          # referencia: DWI
    return (remuestrear_a_grid(flair, grid, sitk.sitkLinear),
            remuestrear_a_grid(adc,   grid, sitk.sitkLinear),
            remuestrear_a_grid(dwi,   grid, sitk.sitkLinear),
            remuestrear_a_grid(mask,  grid, sitk.sitkNearestNeighbor))   # NN: máscara binaria


def preprocesamiento_caso(flair_path, adc_path, dwi_path, mask_path, target_size=(256, 256)):
    flair_i, adc_i, dwi_i, mask_i = alinear_caso(flair_path, adc_path, dwi_path, mask_path)

    flair_a = sitk.GetArrayFromImage(flair_i)
    adc_a   = sitk.GetArrayFromImage(adc_i)
    dwi_a   = sitk.GetArrayFromImage(dwi_i)
    mask_a  = sitk.GetArrayFromImage(mask_i)

    flair_n = z_score_normalize(flair_a)
    adc_n   = z_score_normalize(adc_a)
    dwi_n   = z_score_normalize(dwi_a)

    flair_f = crop_or_pad_volume(flair_n, target_size=target_size)
    adc_f   = crop_or_pad_volume(adc_n,   target_size=target_size)
    dwi_f   = crop_or_pad_volume(dwi_n,   target_size=target_size)
    mask_f  = crop_or_pad_volume(mask_a,  target_size=target_size)

    formas = {tuple(v.shape) for v in (flair_f, adc_f, dwi_f, mask_f)}
    if len(formas) != 1:
        raise RuntimeError(f"Formas distintas tras el preprocesamiento: {formas}")
    return flair_f, adc_f, dwi_f, mask_f


# ------------------------------- CELDA B -------------------------------------
# Diagnóstico visual: ¿el FLAIR remuestreado coincide con el DWI y la lesión?
import matplotlib.pyplot as plt


def diagnostico_alineacion(flair_path, adc_path, dwi_path, mask_path):
    for nombre, p in (("FLAIR", flair_path), ("ADC", adc_path), ("DWI", dwi_path), ("MSK", mask_path)):
        im = sitk.ReadImage(p)
        print(f"{nombre:6s} size={im.GetSize()} spacing={tuple(round(s, 2) for s in im.GetSpacing())} "
              f"origin={tuple(round(o, 1) for o in im.GetOrigin())}")

    flair_i, adc_i, dwi_i, mask_i = alinear_caso(flair_path, adc_path, dwi_path, mask_path)
    f = sitk.GetArrayFromImage(flair_i)
    d = sitk.GetArrayFromImage(dwi_i)
    m = sitk.GetArrayFromImage(mask_i)
    print("Tras alinear (Z, Y, X):", f.shape, d.shape, m.shape)

    z = int(np.argmax(m.reshape(m.shape[0], -1).sum(axis=1)))     # corte con más lesión
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(d[z], cmap="gray");  ax[0].set_title(f"DWI (corte {z})")
    ax[1].imshow(f[z], cmap="gray");  ax[1].set_title("FLAIR remuestreado a la rejilla DWI")
    ax[2].imshow(f[z], cmap="gray");  ax[2].contour(m[z] > 0, colors="r", linewidths=1)
    ax[2].set_title("FLAIR + contorno de la lesión")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def rutas_sujeto(layout, sid):
    """sid SIN el prefijo 'sub-' (p. ej. 'strokecase0062'). Mismas consultas del bucle principal."""
    fl = layout.get(subject=sid, suffix="FLAIR", extension=".nii.gz")[0].path
    ad = layout.get(subject=sid, suffix="adc",   extension=".nii.gz")[0].path
    dw = layout.get(subject=sid, suffix="dwi",   extension=".nii.gz")[0].path
    mk = layout.get(subject=sid, suffix="msk",   extension=".nii.gz", scope="derivatives")[0].path
    return fl, ad, dw, mk


# Uso (después de crear 'layout' como en el notebook original):
# for sid in ["strokecase0062", "strokecase0093", "strokecase0001"]:
#     print("\n=====", sid)
#     diagnostico_alineacion(*rutas_sujeto(layout, sid))


# ------------------------------- CELDA C -------------------------------------
# Reemplaza la celda del bucle principal. Cambios: carpeta de salida NUEVA (v2),
# se omiten sujetos ya procesados (se puede relanzar tras una desconexión) y se
# guarda en float16 (imágenes) / uint8 (máscara).
import glob
import json
from sklearn.model_selection import train_test_split


def procesar_dataset(layout, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subjects = layout.get_subjects()
    print(f"Total de sujetos encontrados: {len(subjects)}")

    for sub_id in subjects:
        nombres = {"FLAIR": f"sub-{sub_id}_FLAIR_proc.npy", "ADC": f"sub-{sub_id}_ADC_proc.npy",
                   "dwi":   f"sub-{sub_id}_dwi_proc.npy",   "msk": f"sub-{sub_id}_msk_proc.npy"}
        if all(os.path.exists(os.path.join(output_dir, n)) for n in nombres.values()):
            continue                                   # ya procesado (permite relanzar)
        print(f"--- Procesando Sujeto: sub-{sub_id} ---")

        flair_files = layout.get(subject=sub_id, suffix='FLAIR', extension='.nii.gz')
        adc_files   = layout.get(subject=sub_id, suffix='adc',   extension='.nii.gz')
        dwi_files   = layout.get(subject=sub_id, suffix='dwi',   extension='.nii.gz')
        mask_files  = layout.get(subject=sub_id, suffix='msk',   extension='.nii.gz', scope='derivatives')

        if flair_files and adc_files and dwi_files and mask_files:
            f_p, a_p, d_p, m_p = preprocesamiento_caso(flair_files[0].path, adc_files[0].path,
                                                       dwi_files[0].path, mask_files[0].path)
            print(f"  OK | forma final (todas las modalidades): {f_p.shape}")
            np.save(os.path.join(output_dir, nombres["FLAIR"]), f_p.astype(np.float16))
            np.save(os.path.join(output_dir, nombres["ADC"]),   a_p.astype(np.float16))
            np.save(os.path.join(output_dir, nombres["dwi"]),   d_p.astype(np.float16))
            np.save(os.path.join(output_dir, nombres["msk"]),   (m_p > 0.5).astype(np.uint8))
        else:
            print(f"  [ADVERTENCIA] A sub-{sub_id} le falta alguna modalidad. Omitiendo...")

    # --- Partición: MISMO código que antes (misma semilla y mismos nombres => misma partición) ---
    archivos_flair = sorted(glob.glob(os.path.join(output_dir, "*_FLAIR_proc.npy")))
    lista_sujetos = sorted(list(set(os.path.basename(f).split('_flair')[0] for f in archivos_flair)))
    print(f"Total de sujetos guardados en '{output_dir}': {len(lista_sujetos)}")

    sujetos_dev, sujetos_test = train_test_split(lista_sujetos, test_size=0.20, random_state=42, shuffle=True)
    sujetos_train, sujetos_val = train_test_split(sujetos_dev, test_size=0.20, random_state=42, shuffle=True)

    dataset_split = {"seed": 42, "output_dir": output_dir, "total_sujetos": len(lista_sujetos),
                     "train": sujetos_train, "val": sujetos_val, "test": sujetos_test}
    print(f"Train: {len(sujetos_train)} | Val: {len(sujetos_val)} | Test: {len(sujetos_test)}")
    with open(os.path.join(output_dir, "dataset_split.json"), "w") as f:
        json.dump(dataset_split, f, indent=4)


# Uso:
# from google.colab import drive
# drive.mount('/content/drive')
# from bids import BIDSLayout
# layout = BIDSLayout('/content/drive/MyDrive/TFM/ISLES-2022', derivatives=True, validate=False)
# procesar_dataset(layout, '/content/drive/MyDrive/TFM/ISLES-2022_Preprocessed_v2')
