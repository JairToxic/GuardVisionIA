import cv2
import torch
import requests
import numpy as np
import threading
from flask import Flask, render_template, Response
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Verifica si la GPU está disponible y usa CUDA si es posible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")  # Verifica si se está usando CUDA o CPU

# Cargar el modelo YOLOv5 preentrenado (usamos 'yolov5n' para optimizar la velocidad)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device)

# URL del endpoint de Power Automate
power_automate_url = 'https://prod-11.westus.logic.azure.com:443/workflows/fa3bd1c3ba094644a2a143c2399b4f11/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=WtIleEshRLLj_jhsEcCVBwGyqgqmLi2nYunYZktxod4'

# Ruta del video en tu máquina
video_path = './La ciudad!! Gente cruzando la calle maña est mapocho.MTS.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar si el video se ha cargado correctamente
if not cap.isOpened():
    raise Exception("Error: No se puede abrir el archivo de video.")

# Usamos ThreadPoolExecutor para gestionar los hilos de manera más eficiente
executor = ThreadPoolExecutor(max_workers=2)

@app.route('/')
def index():
    return render_template('index.html')

# Función para procesar cada frame de manera asíncrona
def process_frame(frame):
    # Redimensionar el frame para mejorar el rendimiento
    frame_resized = cv2.resize(frame, (640, 640))  # Reducción de tamaño de imagen

    # Realizar la detección de objetos
    results = model(frame_resized)

    # Devolver los resultados de la detección
    return results

# Generar el flujo de frames para la transmisión de video
def gen_frames():
    area_coords = (100, 100, 400, 400)  # Coordenadas de ejemplo del área seleccionada
    detection_time = 0
    alert_sent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Llamar a la inferencia en un hilo separado para no bloquear el hilo principal
        future = executor.submit(process_frame, frame)

        # Obtener el resultado de la inferencia
        results = future.result()

        # Filtrar detecciones y aplicar la lógica de la alarma
        detections = results.xyxy[0]  # [xmin, ymin, xmax, ymax, conf, class]
        person_detected_in_area = False

        for *box, conf, cls in detections.tolist():
            cls = int(cls)
            if cls == 0 and conf > 0.7:  # '0' es la clase para 'person' en COCO
                if area_coords[0] < box[0] < area_coords[2] and area_coords[1] < box[1] < area_coords[3]:
                    person_detected_in_area = True
                    detection_time += 1 / cap.get(cv2.CAP_PROP_FPS)

                    if detection_time >= 3 and not alert_sent:
                        requests.post(power_automate_url)
                        alert_sent = True
                    color = (0, 0, 255) if detection_time >= 3 else (0, 255, 0)
                else:
                    color = (0, 255, 0)
                    detection_time = 0

                # Dibujar la caja de detección en el video
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

        # Dibujar el área seleccionada en el video (rectángulo de detección)
        cv2.rectangle(frame, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), (255, 0, 0), 2)

        # Mostrar el tiempo en el video
        cv2.putText(frame, f'Tiempo en zona: {detection_time:.2f} s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convertir el frame a JPEG y devolverlo para la transmisión
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
