!pip install ultralytics
from ultralytics import RTDETR
# 1. Cargar el modelo RT-DETR pre-entrenado (versión 'l' de Large o 'x' de Extra-large)
model_detr = RTDETR('rtdetr-l.pt')
# El entrenamiento de Transformers suele requerir un learning rate más bajo
model_detr.train(
    data='config.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.0001  # Ajuste sugerido para mayor estabilidad en Transformers
)
# 3. Inferencia (Generar el video marcado)
model_detr.predict(source='3.avi', save=True, conf=0.25)