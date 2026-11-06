# -*- coding: utf-8 -*-
"""
utils_training.py

Motor de entrenamiento COMÚN para las 4 arquitecturas (U-Net, V-Net, SegNet, Attention U-Net).
Se llama desde el cuaderno de cada arquitectura; solo cambian el modelo, el optimizador y los loaders.

Novedades respecto a la versión provisional de 01_UNet:
  - Checkpoint al final de CADA época en Drive (modelo + optimizador + historial + early stopping).
  - Reanudación automática: si Colab se desconecta, se vuelve a ejecutar la celda y continúa
    desde la última época completada.
  - Historial guardado en JSON (history_<modelo>.json) para las gráficas del TFM.
  - Precisión mixta (AMP) opcional. IMPORTANTE: usa el mismo valor en las 4 arquitecturas.
  - val_forward opcional: para V-Net 3D se puede pasar una inferencia con ventana deslizante.
  - Aumento de datos en GPU (aumento_gpu): mismos parámetros del anteproyecto que antes se aplicaban
    en la CPU con MONAI (la CPU de Colab, con 2 núcleos, dejaba la GPU al ~8 % de uso).
"""

import os
import time
import json
import math
import torch
import torch.nn.functional as F
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric


def _guardar_atomico(obj, ruta):
    """Escribe en un temporal y renombra: nunca queda un checkpoint a medias en Drive."""
    tmp = ruta + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, ruta)


# ==============================================================================
# AUMENTO DE DATOS EN GPU (2D)
# Mismos parámetros que train_transforms de data_loader.py, pero por lote y en GPU:
#   rotación ±10° (p=0.5) | traslación ≤12 px y escala ±10 % (p=0.3) |
#   flip lateral (p=0.5)  | ruido gaussiano std=0.05 (p=0.3)
# Cada muestra del lote recibe parámetros aleatorios independientes.
# Imagen: interpolación bilineal; máscara: vecino más próximo (sigue siendo 0/1).
# ==============================================================================
@torch.no_grad()
def aumento_gpu(x, y, rot_deg=10.0, p_rot=0.5, p_aff=0.3, trans_px=12.0, escala=0.1,
                p_flip=0.5, p_ruido=0.3, std_ruido=0.05):
    """x: (B, C, H, W) imágenes float32; y: (B, 1, H, W) máscara {0,1}. Devuelve (x, y) aumentados."""
    x = x.as_tensor() if hasattr(x, "as_tensor") else x      # quitar MetaTensor de MONAI
    y = y.as_tensor() if hasattr(y, "as_tensor") else y
    B, _, H, W = x.shape
    dev = x.device

    def u(n=B):                                               # uniforme en [-1, 1]
        return torch.rand(n, device=dev) * 2 - 1

    hay_rot = (torch.rand(B, device=dev) < p_rot).float()
    hay_aff = (torch.rand(B, device=dev) < p_aff).float()
    hay_flip = torch.rand(B, device=dev) < p_flip

    ang = hay_rot * u() * math.radians(rot_deg)
    sc = 1.0 + hay_aff * u() * escala
    tx = hay_aff * u() * trans_px / (W / 2.0)                 # píxeles -> coordenadas normalizadas [-1, 1]
    ty = hay_aff * u() * trans_px / (H / 2.0)
    fl = torch.where(hay_flip, -torch.ones(B, device=dev), torch.ones(B, device=dev))

    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos / sc * fl
    theta[:, 0, 1] = -sin / sc
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / sc * fl
    theta[:, 1, 1] = cos / sc
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    hay_ruido = (torch.rand(B, 1, 1, 1, device=dev) < p_ruido).float()
    x = x + hay_ruido * torch.randn_like(x) * std_ruido
    return x, y


def verificar_aumento(train_loader, device, n=4):
    """Muestra n muestras aumentadas (canal FLAIR) con el contorno de la máscara: DEBEN seguir alineadas."""
    import matplotlib.pyplot as plt
    images, masks = next(iter(train_loader))
    images, masks = aumento_gpu(images.to(device), masks.to(device))
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))
    for i in range(n):
        ax[i].imshow(images[i, 2].cpu(), cmap="gray")            # canal 2 = FLAIR
        if masks[i, 0].sum() > 0:
            ax[i].contour(masks[i, 0].cpu() > 0.5, colors="r", linewidths=1)
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()
    print("valores únicos de la máscara:", torch.unique(masks).tolist())


class EarlyStopping:
    def __init__(self, patience, min_delta=1e-4, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            _guardar_atomico(model.state_dict(), self.save_path)    # mejor modelo (solo pesos)
            print(f" -> Pérdida de validación mejoró. Modelo guardado en: {self.save_path}")
        else:
            self.counter += 1
            print(f" -> Sin mejora en validación. Paciencia: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def state_dict(self):
        return {"counter": self.counter, "best_loss": self.best_loss, "early_stop": self.early_stop}

    def load_state_dict(self, d):
        self.counter = d["counter"]
        self.best_loss = d["best_loss"]
        self.early_stop = d["early_stop"]


def train_network(model, train_loader, val_loader, optimizer, epochs, device, model_name,
                  save_dir, patience=12, use_amp=True, val_forward=None, resume=True, aug_gpu=True):
    """
    save_dir    : carpeta EN DRIVE donde se guardan best_/last_/history_ (p. ej. /content/drive/MyDrive/TFM/resultados/UNet_2D)
    use_amp     : precisión mixta en GPU (más rápido en T4). Igual para las 4 arquitecturas.
    val_forward : función (model, x) -> logits para validar. Por defecto model(x).
    resume      : si existe last_<modelo>.pth, continúa desde ahí.
    aug_gpu     : aplica aumento_gpu a cada lote de TRAIN (usa crear_dataloaders(..., aug_en_gpu=True)).
    """
    os.makedirs(save_dir, exist_ok=True)
    ruta_best = os.path.join(save_dir, f"best_{model_name}.pth")
    ruta_last = os.path.join(save_dir, f"last_{model_name}.pth")
    ruta_hist = os.path.join(save_dir, f"history_{model_name}.json")

    criterion = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0, gamma=2.0)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    model = model.to(device)
    early_stopper = EarlyStopping(patience=patience, save_path=ruta_best)

    device_type = torch.device(device).type
    amp_on = bool(use_amp) and device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    forward_val = val_forward if val_forward is not None else (lambda m, x: m(x))

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "epoch_time": []}
    start_epoch = 1

    # ---------------- Reanudación ----------------
    if resume and os.path.exists(ruta_last):
        ckpt = torch.load(ruta_last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if amp_on and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        early_stopper.load_state_dict(ckpt["early_stopper"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"↻ Reanudando {model_name} desde la época {start_epoch} (última completada: {ckpt['epoch']})")
        if early_stopper.early_stop:
            print("   El entrenamiento ya había terminado por parada temprana.")
            start_epoch = epochs + 1

    print(f"=== ENTRENAMIENTO: {model_name} | {device} | AMP={amp_on} | aug_gpu={aug_gpu} | épocas {start_epoch}-{epochs} ===")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # ---------------- TRAIN ----------------
        model.train()
        running_train_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            if aug_gpu:
                images, masks = aumento_gpu(images, masks)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_on):
                outputs = model(images)
            loss = criterion(outputs.float(), masks)          # la pérdida siempre en float32
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item() * images.size(0)

            if (i + 1) % 200 == 0:
                print(f"Época [{epoch}/{epochs}] - Progreso: {i+1}/{len(train_loader)} lotes procesados")

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # ---------------- VALIDACIÓN ----------------
        model.eval()
        running_val_loss = 0.0
        dice_metric.reset()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_on):
                    outputs = forward_val(model, images)
                outputs = outputs.float()
                running_val_loss += criterion(outputs, masks).item() * images.size(0)
                dice_metric(y_pred=(torch.sigmoid(outputs) > 0.5).float(), y=masks)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_dice = dice_metric.aggregate().item()
        elapsed = time.time() - t0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_dice"].append(epoch_val_dice)
        history["epoch_time"].append(elapsed)

        print(f"Época [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) | Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Dice: {epoch_val_dice:.4f}")

        early_stopper(epoch_val_loss, model)

        # ---------------- CHECKPOINT (cada época) ----------------
        _guardar_atomico({"epoch": epoch,
                          "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict(),
                          "early_stopper": early_stopper.state_dict(),
                          "history": history}, ruta_last)
        with open(ruta_hist, "w") as f:
            json.dump(history, f, indent=2)

        if early_stopper.early_stop:
            print(f"\n[!] Parada temprana activada en la época {epoch}. Entrenamiento finalizado.")
            break

    # Cargar los mejores pesos antes de retornar
    model.load_state_dict(torch.load(ruta_best, map_location=device))
    return model, history
