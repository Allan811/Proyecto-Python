#!/usr/bin/python3
import cv2
import numpy as np
import imutils


def cuatro_camaras():
    captura = cv2.VideoCapture(0)

    while True:
        ret, frame = captura.read()

        ancho = int(captura.get(3))
        alto = int(captura.get(4))

        azul = np.full((alto//2, ancho//2, 3), (80, 0, 0), np.uint8)
        rojo = np.full((alto//2, ancho//2, 3), (0, 0, 80), np.uint8)
        verde = np.full((alto//2, ancho//2, 3), (0, 40, 0), np.uint8)

        fondo = np.zeros(frame.shape, np.uint8)
        parte_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        fondo[:alto//2, :ancho//2] = cv2.rotate(
                                                cv2.add(parte_frame, rojo),
                                                cv2.cv2.ROTATE_180
                                                )

        fondo[alto//2:, :ancho//2] = cv2.add(parte_frame, verde)
        fondo[:alto//2, ancho//2:] = cv2.rotate(
                                                cv2.add(parte_frame, azul),
                                                cv2.cv2.ROTATE_180
                                                )

        fondo[alto//2:, ancho//2:] = parte_frame

        cv2.imshow('Label', fondo)

        if cv2.waitKey(1) == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()


def cuadro():
    captura = cv2.VideoCapture(0)

    imagen = cv2.imread('Cuadroalto.png',
                        cv2.IMREAD_UNCHANGED)
    identificador_cara = cv2.CascadeClassifier(cv2.data.haarcascades +
                                               'haarcascade_frontal'
                                               'face_default.xml')

    while True:
        ret, frame = captura.read()
        caras = identificador_cara.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in caras:
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imagen_redimensionada = imutils.resize(imagen, width=w)
            filas_imagen = imagen_redimensionada.shape[0]

            posicion = frame[
                            y-25:y + filas_imagen - 25,
                            x: x + w
                            ]
            mascara = imagen_redimensionada[:, :, 3]

            mascara_invertida = cv2.bitwise_not(mascara)
            fondo_negro = cv2.bitwise_and(
                                          imagen_redimensionada,
                                          imagen_redimensionada,
                                          mask=mascara
                                          )
            fondo_negro = fondo_negro[:, :, 0:3]
            parte_clara = cv2.bitwise_and(
                                          posicion, posicion,
                                          mask=mascara_invertida
                                          )

            suma = cv2.add(fondo_negro, parte_clara)

            frame[y - 25: y + filas_imagen - 25,
                  x: x + w] = suma

            frame[:filas_imagen, :w] = suma

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    captura.release()
    cv2.destroyAllWindows()


def filtros(entrada):
    captura = cv2.VideoCapture(0)

    file = ''
    divisor = 0

    if entrada == 'Pizzas':
        file = 'Daco_387016.png'
        divisor = 3

    elif entrada == 'Lazo':
        file = 'ribbon-gd866137cc_640.png'
        divisor = 2

    imagen = cv2.imread(file,
                        cv2.IMREAD_UNCHANGED)
    identificador_cara = cv2.CascadeClassifier(cv2.data.haarcascades +
                                               'haarcascade_frontal'
                                               'face_default.xml')

    while True:
        ret, frame = captura.read()
        caras = identificador_cara.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in caras:
            imagen_redimensionada = imutils.resize(imagen, width=w)
            filas_imagen = imagen_redimensionada.shape[0]

            if y - filas_imagen >= 0:

                posicion = frame[y - filas_imagen + filas_imagen//divisor:
                                 y + filas_imagen//divisor, x: x + w]
                mascara = imagen_redimensionada[:, :, 3]

                mascara_invertida = cv2.bitwise_not(mascara)
                fondo_negro = cv2.bitwise_and(
                                              imagen_redimensionada,
                                              imagen_redimensionada,
                                              mask=mascara
                                              )
                fondo_negro = fondo_negro[:, :, 0:3]
                parte_clara = cv2.bitwise_and(
                                              posicion, posicion,
                                              mask=mascara_invertida
                                              )

                suma = cv2.add(fondo_negro, parte_clara)

                frame[y - filas_imagen + filas_imagen//divisor:
                      y + filas_imagen//divisor,
                      x: x + w] = suma

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    captura.release()
    cv2.destroyAllWindows()


def reconocimiento():
    captura = cv2.VideoCapture(0)

    identificador_cara = cv2.CascadeClassifier(
                                               cv2.data.haarcascades +
                                               'haarcascade_frontalface_'
                                               'default.xml'
                                               )
    identificador_ojos = cv2.CascadeClassifier(
                                               cv2.data.haarcascades +
                                               'haarcascade_eye.xml'
                                               )

    while True:
        ret, frame = captura.read()
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = identificador_cara.detectMultiScale(frame_gris, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            reg_gris = frame_gris[y:y+w, x:x+w]
            reg_color = frame[y:y+h, x:x+w]
            ojos = identificador_ojos.detectMultiScale(reg_gris, 1.3, 5)
            for (ex, ey, ew, eh) in ojos:
                cv2.rectangle(
                              reg_color, (ex, ey),
                              (ex + ew, ey + eh), (0, 255, 0), 5
                              )

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()
