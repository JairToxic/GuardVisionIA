import cv2
import torch
import numpy as np
import time

# Cargar el modelo YOLOv5 preentrenado
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)  # Usa la cámara por defecto

# Verificar si la cámara se ha abierto correctamente
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

# Variables para dibujar el rectángulo
drawing = False
start_point = (-1, -1)
area_coords = (0, 0, 0, 0)
area_selected = False

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

# Crear una ventana para mostrar el frame inicial
cv2.namedWindow('Seleccionar Área')
cv2.setMouseCallback('Seleccionar Área', draw_rectangle)

# Mostrar el mensaje de instrucciones hasta que se seleccione un área
while not area_selected:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer un frame de la cámara.")
        break

    cv2.putText(frame, 'Selecciona un área:', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if start_point != (-1, -1) and not drawing:
        cv2.rectangle(frame, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), (0, 255, 0), 2)

    cv2.imshow('Seleccionar Área', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow('Seleccionar Área')

# Diccionario para rastrear el tiempo que cada persona ha estado sentada
person_times = {}
person_entry_times = {}

# Crear el MultiTracker
if hasattr(cv2, 'legacy'):
    trackers = cv2.legacy.MultiTracker()
else:
    trackers = cv2.MultiTracker()

# Obtener los FPS de la cámara
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30  # Establecer FPS predeterminado si no se puede obtener de la cámara
delay = int(1000 / fps)  # Calcular delay en milisegundos

# Procesar el video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer un frame de la cámara.")
        break

    # Cada ciertos frames, actualizar los trackers
    if frame_count % 10 == 0:
        # Reiniciar trackers
        if hasattr(cv2, 'legacy'):
            trackers = cv2.legacy.MultiTracker()
        else:
            trackers = cv2.MultiTracker()

        # Crear una máscara para el área seleccionada
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), 255, -1)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Hacer la detección en el cuadro enmascarado
        results = model(masked_frame)
        detections = results.xyxy[0]  # [xmin, ymin, xmax, ymax, conf, class]

        for *box, conf, cls in detections.tolist():
            cls = int(cls)
            conf = float(conf)

            if cls == 0 and conf > 0.7:
                x_min, y_min, x_max, y_max = map(int, box)
                x_min += area_coords[0]
                x_max += area_coords[0]
                y_min += area_coords[1]
                y_max += area_coords[1]

                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                # Crear un nuevo tracker para cada detección
                if hasattr(cv2, 'legacy'):
                    tracker = cv2.legacy.TrackerCSRT_create()
                else:
                    tracker = cv2.TrackerCSRT_create()
                trackers.add(tracker, frame, bbox)

                # Asignar un ID único
                person_id = len(person_times) + 1
                if person_id not in person_entry_times:
                    person_entry_times[person_id] = time.time()

    else:
        # Actualizar trackers existentes
        success, boxes = trackers.update(frame)

        for i, new_box in enumerate(boxes):
            x, y, w, h = map(int, new_box)
            person_id = i + 1

            # Verificar si la persona está dentro del área definida
            if (area_coords[0] < x < area_coords[2] and area_coords[1] < y < area_coords[3]):
                if person_id not in person_entry_times:
                    person_entry_times[person_id] = time.time()

                elapsed_time = time.time() - person_entry_times[person_id]
                person_times[person_id] = elapsed_time

                # Dibujar la caja y el ID
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {person_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                if person_id in person_entry_times:
                    del person_entry_times[person_id]

    # Dibujar el área seleccionada
    cv2.rectangle(frame, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), (255, 0, 0), 2)
    cv2.putText(frame, 'Área de Detección', (area_coords[0], area_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 1)

    # Mostrar los tiempos de cada persona
    y_offset = 30
    for person_id, elapsed in person_times.items():
        cv2.putText(frame, f'Persona {person_id}: {int(elapsed)}s', (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        y_offset += 30

    # Mostrar el cuadro con la detección
    cv2.imshow('Detección', frame)

    frame_count += 1

    # Esperar y salir si se presiona 'q'
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
