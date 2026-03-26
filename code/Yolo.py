!pip install ultralytics

from ultralytics import YOLO

# 2. Cargar modelo base (Transfer Learning)
model = YOLO('yolov8n.pt')
#model = YOLO('yolo26n.pt') Para usar Yolo26

# 3. Entrenar
# Esto creará una carpeta 'runs/detect/train/weights/best.pt'
model.train(data='config.yaml', epochs=50, imgsz=640)

# 4. CARGAR MODELO ENTRENADO (Paso clave)
model = YOLO('runs/detect/train/weights/best.pt')

# 5. Procesar el video y generar el output marcado 
results = model.predict(
    source='4.avi',
    save=True,
    conf=0.1,
    line_width=2,       # Grosor de la línea del cuadro
    show_labels=True,   # Mostrar el nombre (Pez/Agua)
    show_conf=False,    # Ocultar el número de probabilidad (limpia la imagen)
)