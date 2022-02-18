import cv2
import numpy as np

captura = cv2.VideoCapture(0)

while True:
    ret, frame = captura.read()

    ancho = int(captura.get(3))
    alto = int(captura.get(4))

    azul  = np.full((alto//2, ancho//2, 3), (100, 0, 0), np.uint8)
    rojo = np.full((alto//2, ancho//2, 3), (0, 0, 100), np.uint8)
    verde = np.full((alto//2, ancho//2, 3), (0, 100, 0), np.uint8)


    fondo = np.zeros(frame.shape, np.uint8)
    parte_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    fondo[:alto//2, :ancho//2] = cv2.add(parte_frame, rojo)
    fondo[alto//2:, :ancho//2] = cv2.add(parte_frame, verde)
    fondo[:alto//2, ancho//2:] = cv2.add(parte_frame, azul)
    fondo[alto//2:, ancho//2:] = parte_frame

    cv2.imshow('Label', fondo)

    if cv2.waitKey(1) == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
