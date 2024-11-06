# admin/area_selector.py
import cv2
import numpy as np

def select_area(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el primer frame.")
        return None
    drawing = False
    start_point = (-1, -1)
    area_coords = (0, 0, 0, 0)
    area_selected = False

    def draw_rectangle(event, x, y, flags, param):
        nonlocal start_point, drawing, area_coords, area_selected
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            area_coords = (start_point[0], start_point[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            area_coords = (start_point[0], start_point[1], x, y)
            area_selected = True

    cv2.namedWindow('Seleccionar Área')
    cv2.setMouseCallback('Seleccionar Área', draw_rectangle)
    while not area_selected:
        cv2.imshow('Seleccionar Área', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Seleccionar Área')
    cap.release()
    return area_coords if area_selected else None
