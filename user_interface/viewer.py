# user_interface/viewer.py
import cv2

def display_video(video_path, model, area_coords, power_automate_url):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    delay = int(100 / fps)
    detection_time = 0
    alert_sent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al cargar.")
            break

        person_detected, detection_time, alert_sent = detect_persons(
            model, frame, area_coords, fps, power_automate_url=power_automate_url
        )
        
        color = (0, 0, 255) if alert_sent else (0, 255, 0)
        cv2.rectangle(frame, (area_coords[0], area_coords[1]), (area_coords[2], area_coords[3]), color, 2)
        cv2.putText(frame, f'Tiempo: {int(detection_time)}s', (30, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Detección', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
