# -*- coding: utf-8 -*-
"""
segnet.py

MONAI no incluye SegNet (Badrinarayanan et al., 2017), así que se implementa aquí.
Rasgo distintivo de SegNet frente a U-Net/Attention U-Net: el decodificador NO usa
skip connections con las activaciones del codificador; en su lugar reutiliza los
ÍNDICES del max-pooling (MaxUnpool2d) para reposicionar cada valor exactamente
donde estaba antes de reducir la resolución.

Perfil de canales y downsampling IGUAL al de la U-Net del TFM
(channels=(16,32,64,128,256), strides=(2,2,2,2)) para que la comparación de
capacidad entre arquitecturas sea justa: 4 reducciones (x16 en total) y el
mismo ancho de canal en cada nivel.
"""

import torch
import torch.nn as nn


class _BloqueConv(nn.Module):
    """conv 3x3 -> BatchNorm -> ReLU, repetido n_convs veces (clásico de SegNet)."""

    def __init__(self, in_ch, out_ch, n_convs=2):
        super().__init__()
        capas = []
        for i in range(n_convs):
            c_in = in_ch if i == 0 else out_ch
            capas += [nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(inplace=True)]
        self.bloque = nn.Sequential(*capas)

    def forward(self, x):
        return self.bloque(x)


class SegNet(nn.Module):
    """
    SegNet 2D. Firma de llamada similar a monai.networks.nets.UNet/AttentionUnet
    para poder reutilizar el mismo patrón de celda en los 4 cuadernos.

    in_channels, out_channels : igual que en los otros 3 modelos (3 canales de
        entrada -> DWI/ADC/FLAIR, 1 canal de salida -> logits de la lesión).
    channels : anchos de canal por nivel, de superficial a profundo. Por defecto
        el mismo perfil que la U-Net del TFM.
    """

    def __init__(self, in_channels=3, out_channels=1, channels=(16, 32, 64, 128, 256)):
        super().__init__()
        assert len(channels) >= 2, "Se necesitan al menos 2 niveles de canal."
        self.n_pool = len(channels) - 1     # nº de reducciones = nº de niveles - 1 (bottleneck sin pool)

        # ---------------- Codificador ----------------
        self.enc = nn.ModuleList()
        c_in = in_channels
        for c_out in channels:
            self.enc.append(_BloqueConv(c_in, c_out))
            c_in = c_out
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # ---------------- Decodificador (simétrico, en orden inverso) ----------------
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec = nn.ModuleList()
        canales_dec = list(reversed(channels))          # p.ej. 256,128,64,32,16
        for i in range(len(canales_dec) - 1):
            self.dec.append(_BloqueConv(canales_dec[i], canales_dec[i + 1]))

        self.clasificador = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        indices, tamanos = [], []

        # Codificador: bloque conv en cada nivel salvo el último (bottleneck),
        # con max-pool + guardado de índices entre niveles. Los índices de cada
        # pool tienen el MISMO número de canales que la activación justo antes
        # de reducir resolución (p.ej. el pool tras el nivel de 128 canales
        # guarda índices de 128 canales, no de 256).
        for i, bloque in enumerate(self.enc):
            x = bloque(x)
            if i < self.n_pool:
                tamanos.append(x.shape[-2:])
                x, ind = self.pool(x)
                indices.append(ind)

        # Decodificador: en cada paso, PRIMERO se reduce el nº de canales con
        # la conv (para igualar el nº de canales de los índices que se van a
        # usar) y LUEGO se hace el unpool. Si el orden se invierte, el unpool
        # recibe una activación con más canales de los que tienen los índices
        # guardados y PyTorch lanza RuntimeError (como en el traceback).
        for bloque in self.dec:
            x = bloque(x)
            ind = indices.pop()
            size = tamanos.pop()
            x = self.unpool(x, ind, output_size=size)

        return self.clasificador(x)


if __name__ == "__main__":
    # Prueba rápida de formas y de tamaños de lote atípicos (no valida
    # entrenamiento, solo el flujo del tensor y la contabilidad de índices).
    for canales in [(16, 32, 64, 128, 256), (16, 32, 64), (32, 64, 128, 256)]:
        m = SegNet(in_channels=3, out_channels=1, channels=canales)
        for H, W in [(256, 256), (144, 224)]:      # 144x224: tamaño real tras el remuestreo v2
            x = torch.randn(2, 3, H, W)
            y = m(x)
            assert y.shape == (2, 1, H, W), (canales, H, W, y.shape)
    n_params = sum(p.numel() for p in SegNet().parameters())
    print("OK: formas correctas en todos los casos | parámetros (perfil por defecto):", f"{n_params:,}")
