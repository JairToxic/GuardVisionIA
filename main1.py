import cv2
import torch
import numpy as np
from sort import Sort  # Importa el módulo SORT para el seguimiento de objetos

# Configuración para CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Cargar el modelo YOLOv5 preentrenado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Ruta del video
video_path = './La ciudad!! Gente cruzando la calle maña est mapocho.MTS.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar si el video se ha cargado correctamente
if not cap.isOpened():
    print("Error: No se puede abrir el archivo de video.")
    exit()

# Configuración inicial para el área de detección y variables de seguimiento
drawing = False
start_point = (-1, -1)
area_coords = (0, 0, 0, 0)
area_selected = False

# Inicializar el objeto SORT para el seguimiento
tracker = Sort()  # Inicializa el tracker SORT

# Función para dibujar el área seleccionada
def draw_rectangle(event, x, y, flags, param):
    global start_point, drawing, area_coords, area_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            area_coords = (start_point[0], start_point[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        area_coords = (start_point[0], start_point[1], x, y)
        area_selected = True

cv2.namedWindow('Seleccionar Área')
cv2.setMouseCallback('Seleccionar Área', draw_rectangle)

# Seleccionar área en el primer cuadro
ret, frame = cap.read()
if not ret:
    print("Error: No se pudo leer el primer frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while not area_selected:
    temp_frame = frame.copy()
    cv2.putText(temp_frame, 'Selecciona un área:', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if start_point != (-1, -1) and not drawing:
        cv2.rectangle(temp_frame, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), (0, 255, 0), 2)
    cv2.imshow('Seleccionar Área', temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow('Seleccionar Área')

# Reiniciar el video al inicio
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
delay = int(1000 / fps)

# Contador único de personas detectadas
unique_people_count = set()

# Detección y seguimiento de personas
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al cargar.")
        break

    # Aplicar máscara para el área seleccionada
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), 255, -1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Detección de objetos
    results = model(masked_frame)
    detections = results.xyxy[0]

    # Convertir detecciones al formato esperado por SORT
    dets = []
    for *box, conf, cls in detections.tolist():
        if int(cls) == 0 and conf > 0.7:  # Clase 0 es 'persona' en YOLOv5 COCO
            x1, y1, x2, y2 = map(int, box[:4])
            dets.append([x1, y1, x2, y2, conf])

    dets = np.array(dets)
    
    # Aplicar el tracker de SORT y obtener IDs únicos
    tracked_objects = tracker.update(dets)

    for x1, y1, x2, y2, object_id in tracked_objects:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = (0, 255, 0)  # Verde para indicar detección

        # Dibujar la caja y el identificador de la persona
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {int(object_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Añadir el ID único de la persona a nuestro conjunto
        unique_people_count.add(int(object_id))

    # Mostrar el contador de personas únicas
    cv2.putText(frame, f'Personas únicas detectadas: {len(unique_people_count)}', (30, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar el cuadro con las detecciones
    cv2.imshow('Detección y Seguimiento', frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
