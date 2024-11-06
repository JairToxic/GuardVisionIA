# video_processing/detector.py
import torch
import requests

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    print("Modelo YOLO cargado exitosamente.")
    return model

def detect_persons(model, frame, area_coords, fps, alert_threshold=3, power_automate_url=None):
    # Crear una máscara para el área seleccionada
    mask = torch.zeros(frame.shape[:2], dtype=torch.uint8)
    mask[area_coords[1]:area_coords[3], area_coords[0]:area_coords[2]] = 1
    masked_frame = torch.bitwise_and(frame, frame, mask=mask)

    results = model(masked_frame)
    detections = results.xyxy[0]
    person_detected_in_area = False
    detection_time = 0  # Contador de tiempo en segundos
    alert_sent = False

    for *box, conf, cls in detections.tolist():
        cls = int(cls)
        if cls == 0 and conf > 0.7:  # Clase '0' es para 'person' en COCO
            box[0] += area_coords[0]
            box[1] += area_coords[1]
            box[2] += area_coords[0]
            box[3] += area_coords[1]
            if (area_coords[0] < box[0] < area_coords[2] and 
                area_coords[1] < box[1] < area_coords[3]):
                person_detected_in_area = True
                detection_time += 1 / fps
                if detection_time >= alert_threshold and not alert_sent:
                    if power_automate_url:
                        requests.post(power_automate_url)
                    alert_sent = True
    return person_detected_in_area, detection_time, alert_sent
