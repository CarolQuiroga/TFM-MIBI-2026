# TFM-MIBI-2026
Estudio comparativo de arquitecturas de Redes Neuronales Convolucionales (CNN) para la segmentación automatizada de lesiones por ACV isquémico agudo en imágenes de resonancia magnética multisecuencia (DWI, ADC y FLAIR)


Entorno de ejecución. Los experimentos se realizaron en Google Colab con una GPU NVIDIA T4 (16 GB), Python 3.13, PyTorch 2.11 y MONAI 1.6.0. Otras dependencias: SimpleITK, PyBIDS, NumPy, SciPy, pandas, scikit-learn y Matplotlib. [Si incluyes un archivo requirements.txt, menciónalo aquí.]

Datos. El conjunto de datos ISLES 2022 no se incluye en el repositorio. Puede descargarse de su repositorio oficial (Hernandez Petzsche et al., 2022b) bajo licencia CC BY 4.0. La partición de pacientes utilizada (entrenamiento, validación y test) se proporciona en el archivo dataset_split.json.

Estructura del repositorio

Archivo	Función
preprocesamiento_v2.py	Reorientación, remuestreo a una rejilla común de 1 mm, normalización z-score, recorte o relleno a 256 × 256 y partición por paciente (apartado 4.2).
data_loader.py	Construcción de las entradas 2D (DWI, ADC y FLAIR), selección de cortes y cargadores de datos (apartado 4.3).
utils_training.py	Bucle de entrenamiento común: aumento de datos en GPU, función de pérdida DiceFocal, precisión mixta, parada temprana y guardado de puntos de control (apartado 4.3).
segnet.py	Implementación propia de SegNet.
vnet2d.py	Envoltorio de V-Net (MONAI) para su uso en 2D.
01_UNet.ipynb – 04_VNet2D.ipynb	Entrenamiento de cada arquitectura.
evaluacion.py / Evaluacion.ipynb	Inferencia sobre el conjunto de test, cálculo de métricas por paciente y análisis estadístico (apartado 4.4).

Orden de ejecución.

-Preprocesamiento.
-Entrenamiento de las cuatro arquitecturas (cuadernos 01 a 04).
-Evaluación.


Reproducibilidad. Se utilizó una semilla fija (42) para la partición de los datos, la selección de cortes, la inicialización de los modelos y el bootstrap. Las reanudaciones del entrenamiento tras desconexiones de Colab pueden introducir pequeñas variaciones respecto a una ejecución continua (véase el apartado 4.3).
