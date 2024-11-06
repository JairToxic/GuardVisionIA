import cv2
import torch
import requests
import numpy as np

# Cargar el modelo YOLOv5 preentrenado
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)  # Usa la cámara por defecto

# Verificar si la cámara se ha abierto correctamente
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
else:
    # Leer un frame inicial de la cámara
    ret, initial_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        cap.release()
    else:
        # Crear una ventana y establecer el callback de mouse para dibujar
        cv2.namedWindow('Vivo')
        
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
                area_selected = False  # Reiniciar selección al hacer un nuevo clic

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    area_coords = (start_point[0], start_point[1], x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                area_coords = (start_point[0], start_point[1], x, y)
                area_selected = True

        # Establecer el callback del mouse
        cv2.setMouseCallback('Vivo', draw_rectangle)

        # Mostrar el frame inicial para la selección del área
        while True:
            frame_copy = initial_frame.copy()

            # Mostrar instrucciones
            cv2.putText(frame_copy, 'Selecciona un área y presiona "c" para continuar', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Dibujar el área seleccionada mientras se arrastra el mouse
            if start_point != (-1, -1) and drawing:
                cv2.rectangle(frame_copy, (start_point[0], start_point[1]), (area_coords[2], area_coords[3]), (0, 255, 0), 2)

            # Dibujar el rectángulo final una vez seleccionado
            if area_selected:
                cv2.rectangle(frame_copy, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), (0, 255, 0), 2)

            # Mostrar el cuadro con el área de selección
            cv2.imshow('Vivo', frame_copy)

            # Continuar solo si el usuario presiona 'c' después de seleccionar el área
            if cv2.waitKey(1) & 0xFF == ord('c') and area_selected:
                break

        # Cerrar la ventana de selección de área
        cv2.destroyWindow('Vivo')

        alert_threshold = 3  # 3 segundos
        detection_time = 0  # Contador de tiempo en segundos
        alert_sent = False
        last_detection_time = 0  # Último tiempo de detección

        # URL del endpoint de Power Automate (mantén tu URL)
        power_automate_url = 'https://prod-11.westus.logic.azure.com:443/workflows/fa3bd1c3ba094644a2a143c2399b4f11/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=WtIleEshRLLj_jhsEcCVBwGyqgqmLi2nYunYZktxod4'

        # Obtener los FPS de la cámara
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30  # Establecer FPS predeterminado si no se puede obtener de la cámara
        delay = int(1000 / fps)  # Calcular delay en milisegundos

        # Iniciar la transmisión en tiempo real con el área seleccionada
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer un frame de la cámara.")
                break

            # Dibujar el área seleccionada en cada frame
            cv2.rectangle(frame, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), (0, 255, 0), 2)
            cv2.putText(frame, 'Reconocimiento en Área', (area_coords[0], area_coords[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Crear una máscara para el área seleccionada
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), 255, -1)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Hacer la detección en el cuadro enmascarado
            results = model(masked_frame)
            detections = results.xyxy[0]  # [xmin, ymin, xmax, ymax, conf, class]

            person_detected_in_area = False

            for *box, conf, cls in detections.tolist():
                cls = int(cls)

                if cls == 0 and conf > 0.7:  # Verificar si es 'persona' con confianza alta
                    # Ajustar las coordenadas de la caja de detección al área seleccionada
                    box[0] += area_coords[0]
                    box[1] += area_coords[1]
                    box[2] += area_coords[0]
                    box[3] += area_coords[1]

                    if (area_coords[0] < box[0] < area_coords[2] and 
                        area_coords[1] < box[1] < area_coords[3]):
                        person_detected_in_area = True
                        detection_time += 1 / fps

                        # Cambiar el color de la caja si supera el umbral de alerta
                        if detection_time >= alert_threshold:
                            color = (0, 0, 255)  # Rojo
                            if not alert_sent:
                                requests.post(power_automate_url)  # Enviar notificación
                                alert_sent = True
                        else:
                            color = (0, 255, 0)  # Verde si no ha alcanzado el tiempo
                    else:
                        color = (0, 255, 0)  # Verde si está fuera del área
                        detection_time = 0  # Reiniciar el contador si sale del área

                    # Dibujar la caja de detección
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(frame, 'Persona', (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Si no hay personas detectadas
            if not person_detected_in_area:
                last_detection_time += 1 / fps
                if last_detection_time >= 3:
                    detection_time = 0
                    alert_sent = False
            else:
                last_detection_time = 0

            # Mostrar el contador de detección
            cv2.putText(frame, f'Tiempo: {int(detection_time)}s', (30, frame.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Mostrar el cuadro con la detección y el área seleccionada, aqui creaba una nueva ventana con otro nombre
            cv2.imshow('Vivo', frame)

            # Esperar y salir si se presiona 'q'
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
