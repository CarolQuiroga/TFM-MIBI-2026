# -*- coding: utf-8 -*-
"""
data_loader.py (versión corregida)

Cambios respecto a la versión original:
  1. Lee SOLO el corte necesario con mmap (antes np.load() cargaba los 4 volúmenes
     3D completos, ~200 MB, por cada corte -> entrenamiento extremadamente lento).
  2. Si falta una modalidad o las formas no coinciden, FALLA con un mensaje claro
     (antes rellenaba con ceros / recortaba el índice z y ocultaba desalineaciones).
  3. Descarta cortes sin cerebro y permite submuestrear cortes sin lesión en train
     (mismo criterio y misma semilla para las 4 arquitecturas => comparación justa).
  4. crear_dataloaders() ahora respeta batch_size / num_workers (antes estaban fijos).
  5. Aumento de datos alineado con el anteproyecto (rotación ±10°, traslación,
     ruido gaussiano) y padding con ceros (antes 'reflection' duplicaba fondo/lesión).
  6. Sin Resized (las imágenes ya son 256x256; la operación era un no-op costoso).

Convención de archivos (float16 para imágenes, uint8 para máscara):
  <id>_FLAIR_proc.npy, <id>_ADC_proc.npy, <id>_dwi_proc.npy, <id>_msk_proc.npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import monai.transforms as MT
from monai.data import list_data_collate

SUFIJO_FLAIR = "_FLAIR_proc.npy"

# ==============================================================================
# TRANSFORMACIONES (MONAI)
# ==============================================================================
train_transforms = MT.Compose([
    MT.RandRotated(
        keys=["image", "mask"],
        range_x=np.deg2rad(10),               # ±10° (anteproyecto)
        prob=0.5,
        mode=["bilinear", "nearest"],         # 'nearest' indispensable para la máscara
        padding_mode="zeros",
    ),
    MT.RandAffined(
        keys=["image", "mask"],
        prob=0.3,
        translate_range=(12, 12),             # ~4.7 % de 256 px (anteproyecto: máx. 10 %)
        scale_range=(0.1, 0.1),               # OJO: no está en el anteproyecto -> documentarlo
        mode=["bilinear", "nearest"],
        padding_mode="zeros",
    ),
    MT.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),  # OJO: tampoco está en el anteproyecto
    MT.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),  # ruido gaussiano (anteproyecto)
    MT.ToTensord(keys=["image", "mask"]),
])

val_transforms = MT.Compose([
    MT.ToTensord(keys=["image", "mask"]),
])


# ==============================================================================
# DATASET
# ==============================================================================
class StrokeDatasetNPY(Dataset):
    """
    Dataset de cortes 2D. Cada item es un corte axial con 3 canales (DWI, ADC, FLAIR).

    lista_rutas_flair : rutas completas a los *_FLAIR_proc.npy (una por sujeto)
    prop_sin_lesion   : fracción de cortes CON cerebro pero SIN lesión que se conserva
                        (1.0 = todos). Se sortea una sola vez con 'seed'.
    """

    def __init__(self, lista_rutas_flair, transform=None, prop_sin_lesion=1.0, seed=42):
        self.transform = transform
        self.indices_slices = []          # (ruta_flair, z)
        rng = np.random.default_rng(seed)
        errores = []
        n_con, n_sin_total, n_sin_usados = 0, 0, 0

        for flair_path in lista_rutas_flair:
            rutas = self._rutas(flair_path)
            nombre = os.path.basename(flair_path)

            faltan = [k for k, p in rutas.items() if not os.path.exists(p)]
            if faltan:
                errores.append(f"{nombre}: faltan {faltan}")
                continue

            vols = {k: np.load(p, mmap_mode="r") for k, p in rutas.items()}
            formas = {k: tuple(v.shape) for k, v in vols.items()}
            if len(set(formas.values())) != 1:
                errores.append(f"{nombre}: formas distintas entre modalidades {formas}")
                continue

            n_z = vols["flair"].shape[0]
            hay_lesion = vols["msk"].reshape(n_z, -1).max(axis=1) > 0
            hay_cerebro = (vols["flair"].reshape(n_z, -1) != 0).any(axis=1) | hay_lesion

            con_lesion = [z for z in range(n_z) if hay_cerebro[z] and hay_lesion[z]]
            sin_lesion = [z for z in range(n_z) if hay_cerebro[z] and not hay_lesion[z]]
            n_con += len(con_lesion)
            n_sin_total += len(sin_lesion)

            if prop_sin_lesion < 1.0 and sin_lesion:
                k = int(round(len(sin_lesion) * prop_sin_lesion))
                sin_lesion = sorted(rng.choice(sin_lesion, size=k, replace=False).tolist()) if k else []
            n_sin_usados += len(sin_lesion)

            for z in sorted(con_lesion + sin_lesion):
                self.indices_slices.append((flair_path, z))

        if errores:
            raise RuntimeError("Problemas en los datos locales:\n  " + "\n  ".join(errores))

        print(f"[Dataset] sujetos={len(lista_rutas_flair)} | cortes con lesión={n_con} | "
              f"sin lesión (con cerebro)={n_sin_total} -> usados={n_sin_usados} | total={len(self.indices_slices)}")

    @staticmethod
    def _rutas(flair_path):
        if not flair_path.endswith(SUFIJO_FLAIR):
            raise ValueError(f"Se esperaba una ruta terminada en '{SUFIJO_FLAIR}': {flair_path}")
        base = flair_path[: -len(SUFIJO_FLAIR)]
        return {
            "flair": base + "_FLAIR_proc.npy",
            "adc":   base + "_ADC_proc.npy",
            "dwi":   base + "_dwi_proc.npy",
            "msk":   base + "_msk_proc.npy",
        }

    @staticmethod
    def _corte(ruta, z):
        # mmap: solo se lee del disco el corte z (no el volumen completo)
        return np.asarray(np.load(ruta, mmap_mode="r")[z], dtype=np.float32)

    def __len__(self):
        return len(self.indices_slices)

    def __getitem__(self, idx):
        flair_path, z = self.indices_slices[idx]
        r = self._rutas(flair_path)

        # Orden de canales idéntico al original: DWI, ADC, FLAIR
        image = np.stack([self._corte(r["dwi"], z),
                          self._corte(r["adc"], z),
                          self._corte(r["flair"], z)], axis=0)          # (3, H, W)
        mask = (self._corte(r["msk"], z) > 0.5).astype(np.float32)[None]  # (1, H, W)

        data = {"image": image, "mask": mask}
        if self.transform:
            data = self.transform(data)
        return data["image"], data["mask"]


# ==============================================================================
# DATALOADERS
# ==============================================================================
def crear_dataloaders(archivos_train, archivos_val, batch_size=16, num_workers=2,
                      prop_sin_lesion_train=0.3, seed=42, aug_en_gpu=True):
    """
    archivos_*: listas de rutas a *_FLAIR_proc.npy
    prop_sin_lesion_train: fracción de cortes sin lesión que se usa en TRAIN.
        Validación usa siempre todos los cortes con cerebro.
        Usa los MISMOS valores para las 4 arquitecturas y documéntalo en la metodología.
    aug_en_gpu: True  -> el loader NO aumenta datos; el aumento (mismos parámetros) lo aplica
                        utils_training.aumento_gpu en la GPU (train_network(..., aug_gpu=True)).
                False -> aumento en CPU con MONAI (train_transforms), mucho más lento en Colab.
    """
    ds_train = StrokeDatasetNPY(archivos_train, transform=(val_transforms if aug_en_gpu else train_transforms),
                                prop_sin_lesion=prop_sin_lesion_train, seed=seed)
    ds_val = StrokeDatasetNPY(archivos_val, transform=val_transforms,
                              prop_sin_lesion=1.0, seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
              collate_fn=list_data_collate,
              # persistent_workers=False: si se interrumpe el entrenamiento (botón stop / desconexión),
              # los workers persistentes quedan muertos y el siguiente intento falla con
              # 'DataLoader worker exited unexpectedly'. Ahora los workers solo leen del disco,
              # así que reiniciarlos en cada época cuesta ~1 s.
              persistent_workers=False)

    train_loader = DataLoader(ds_train, shuffle=True, generator=g, **kw)
    val_loader = DataLoader(ds_val, shuffle=False, **kw)
    return train_loader, val_loader
