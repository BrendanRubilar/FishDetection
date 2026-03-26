!pip install ultralytics

from ultralytics import YOLO

# 2. Cargar modelo base (Transfer Learning) [cite: 311, 347, 582]
model = YOLO('yolov8n.pt')

# 3. Entrenar [cite: 19, 153, 579]
# Esto creará una carpeta 'runs/detect/train/weights/best.pt'
model.train(data='config.yaml', epochs=50, imgsz=640)

# 4. CARGAR TU MODELO ENTRENADO (Paso clave)
# No uses el 'yolov8n.pt' para el video, usa el que acabas de generar
model = YOLO('runs/detect/train/weights/best.pt')


# 5. Procesar el video y generar el output marcado [cite: 79, 188, 591]
results = model.predict(
    source='4.avi',
    save=True,
    conf=0.1,
    line_width=2,       # Grosor de la línea del cuadro
    show_labels=True,   # Mostrar el nombre (Pez/Agua)
    show_conf=False,    # Ocultar el número de probabilidad (limpia la imagen)
)