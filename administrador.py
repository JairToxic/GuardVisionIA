import cv2
import numpy as np

class Administrador:
    def __init__(self):
        self.drawing = False
        self.start_point = (-1, -1)
        self.area_coords = (0, 0, 0, 0)
        self.area_selected = False
        self.alert_thresholds = []

    def seleccionar_area(self, frame):
        """Permite al usuario seleccionar el área en el video."""
        
        def draw_rectangle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.area_coords = (self.start_point[0], self.start_point[1], x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.area_coords = (self.start_point[0], self.start_point[1], x, y)
                self.area_selected = True
        
        cv2.namedWindow('Seleccionar Área')
        cv2.setMouseCallback('Seleccionar Área', draw_rectangle)
        
        while not self.area_selected:
            cv2.putText(frame, 'Selecciona un área:', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Seleccionar Área', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('Seleccionar Área')

    def configurar_alarma(self, primer_alarma, siguiente_alarma):
        """Configura los tiempos de alarma inicial y recurrente."""
        self.alert_thresholds = [primer_alarma, siguiente_alarma]

