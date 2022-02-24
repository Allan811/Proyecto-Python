#!/usr/bin/python3
"""
Modulo con las funciones para ser ejecutadas.
Solo se debe ejecutar el archivo Principal.py.
"""


import cv2
import numpy as np
import imutils


def cuatro_camaras():
    """
    Función para el filtro de cuatro cámaras.
    """

    # Se define el video de entrada.
    captura = cv2.VideoCapture(0)

    while True:
        # Se define la variable frame como la imagen de la cámara.
        ret, frame = captura.read()

        # Se define el ancho y alto de la captura de cámara.
        ancho = int(captura.get(3))
        alto = int(captura.get(4))

        # Se definen las matrices para cambiar color.
        azul = np.full((alto//2, ancho//2, 3), (80, 0, 0), np.uint8)
        rojo = np.full((alto//2, ancho//2, 3), (0, 0, 80), np.uint8)
        verde = np.full((alto//2, ancho//2, 3), (0, 40, 0), np.uint8)

        # Se define la matriz que va a funcionar como fondo.
        fondo = np.zeros(frame.shape, np.uint8)

        # Se redimensiona frame.
        parte_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Se aplican las rotaciones, colores y lugares en donde se va a
        # mostrar cada cámara.
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

        # Se muestra el resultado.
        cv2.imshow('Cuatro_camaras', fondo)

        # Se define la tecla para terminar el proceso.
        if cv2.waitKey(1) == ord('q'):
            break

    # Se libera la cámara y se cierran todas las ventanas.
    captura.release()
    cv2.destroyAllWindows()


def filtros(entrada):
    """
    Función para el filtro de las pizzas y el lazo.
    """
    captura = cv2.VideoCapture(0)

    # Se defiene las variables que cambian dependiendo de la entrada.
    archivo = ''
    divisor = 0

    # Se definen los valores dependiendo de la entrada.
    if entrada == 'Pizzas':
        archivo = 'Daco_387016.png'
        divisor = 3

    elif entrada == 'Lazo':
        archivo = 'ribbon-gd866137cc_640.png'
        divisor = 2

    # Se lee la imagen de entrada.
    imagen = cv2.imread(archivo,
                        cv2.IMREAD_UNCHANGED)
    identificador_cara = cv2.CascadeClassifier(cv2.data.haarcascades +
                                               'haarcascade_frontal'
                                               'face_default.xml')

    while True:
        ret, frame = captura.read()
        caras = identificador_cara.detectMultiScale(frame, 1.3, 5)

        # Se redimensiona la imagen de entrada para que tenga el ancho
        # de la cara que se indentifique.
        for (x, y, w, h) in caras:
            imagen_redimensionada = imutils.resize(imagen, width=w)
            filas_imagen = imagen_redimensionada.shape[0]

            # Se crea un excepción para solo poner la imagen si esta no se
            # sale de la pantalla.
            if y - filas_imagen >= 0:

                # Se define la posición en donde se encontrará la imagen.
                posicion = frame[y - filas_imagen + filas_imagen//divisor:
                                 y + filas_imagen//divisor, x: x + w]

                # Se define la máscara con el canal en la posición 3 de la
                # imagen redimensiona.
                mascara = imagen_redimensionada[:, :, 3]

                # Se invierte la máscara.
                mascara_invertida = cv2.bitwise_not(mascara)

                # Con bitwise_and se genera la imagen con el fondo negro.
                fondo_negro = cv2.bitwise_and(
                                              imagen_redimensionada,
                                              imagen_redimensionada,
                                              mask=mascara
                                              )

                # Se utiliza todo el fondo negro pero con solo los primeros 3
                # canales.
                fondo_negro = fondo_negro[:, :, 0:3]

                # cv2.imshow('Fondo negro', fondo_negro)

                # Se define la imagen con el fondo del video y la imagen en
                # negro.
                parte_clara = cv2.bitwise_and(
                                              posicion, posicion,
                                              mask=mascara_invertida
                                              )

                # cv2.imshow('Parte clara', parte_clara)
                # Se suman para dar el resultado final.
                suma = cv2.add(fondo_negro, parte_clara)

                # Se define qué parte  de frame cambiar por el resultado
                # en la variable suma.

                frame[y - filas_imagen + filas_imagen//divisor:
                      y + filas_imagen//divisor,
                      x: x + w] = suma

        # Se muestra frame.
        cv2.imshow('Filtro', frame)

        # Se define 'q' como la tecla para terminar el proceso.
        if cv2.waitKey(1) == ord('q'):
            break
    # Se libera la cámara y se cierran todas las ventanas.
    captura.release()
    cv2.destroyAllWindows()


def cuadro():
    """
    Función para el filtro del cuadro.
    """
    captura = cv2.VideoCapture(0)

    # Se lee la imagen.
    imagen = cv2.imread('Cuadroalto.png',
                        cv2.IMREAD_UNCHANGED)

    # Se especifíca qué identificador se desea utilizar por medio del
    # archivo .xml
    identificador_cara = cv2.CascadeClassifier(cv2.data.haarcascades +
                                               'haarcascade_frontal'
                                               'face_default.xml')

    while True:
        ret, frame = captura.read()

        # Se utiliza detectMultiScale para encontrar las caras.
        caras = identificador_cara.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in caras:

            # Se redimensiona la imagen del cuadro para que tenga el ancho
            # de la cara que se detecte.
            imagen_redimensionada = imutils.resize(imagen, width=w)
            filas_imagen = imagen_redimensionada.shape[0]

            # Se define la posición en donde se mostrará el cuadro.
            posicion = frame[
                            y-25:y + filas_imagen - 25,
                            x: x + w
                            ]

            # Se define la máscara con el canal 3 de la imagen.
            mascara = imagen_redimensionada[:, :, 3]

            # Se invierte la máscara.
            mascara_invertida = cv2.bitwise_not(mascara)

            # Se define la imagen redimensiona con fondo negro.
            fondo_negro = cv2.bitwise_and(
                                          imagen_redimensionada,
                                          imagen_redimensionada,
                                          mask=mascara
                                          )

            # Se toman solo los primeros 3 canales para no entrar en conflicto
            # con frame que solo tiene 3 canales.
            fondo_negro = fondo_negro[:, :, 0:3]

            # Con bitwise_and se define la imagen con fondo del video y la
            # imagen de entrada en negro.
            parte_clara = cv2.bitwise_and(
                                          posicion, posicion,
                                          mask=mascara_invertida
                                          )

            # Se suman para tener el resultado final.
            suma = cv2.add(fondo_negro, parte_clara)

            # Se especifíca en dónde se quiere poner la imagen final.
            frame[y - 25: y + filas_imagen - 25,
                  x: x + w] = suma

            # Se muestra el cuadro con la cara en la esquina superior
            # izquierda.
            frame[:filas_imagen, :w] = suma

        # Se muestra frame.
        cv2.imshow('Cuadro', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()


def reconocimiento():
    """
    Función para el reconocimiento facial y de ojos.
    """

    captura = cv2.VideoCapture(0)

    # Se especifíca el archivo .xml para el identificador de cara y ojos.
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

        # Se utiliza cvtColor para cambiar frame a escala de grises.
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Con detectMultiScale se idenfitica la posicion de las caras
        # en la imagen.
        caras = identificador_cara.detectMultiScale(frame_gris, 1.3, 5)

        # Se hace un for para iterar sobre la tuple que retorna caras.
        for (x, y, w, h) in caras:

            # Con cv2.rectangle se crea un rectángulo sobre los rostros
            # encontrdos.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Se define la region de interés para buscar los ojos en el
            # frame en gris.
            reg_gris = frame_gris[y: y + w, x: x + w]

            # Se define la region de interés para buscar los ojos en el
            # frame a color.
            reg_color = frame[y: y + h, x: x + w]

            # Se identifican los ojos.
            ojos = identificador_ojos.detectMultiScale(reg_gris, 1.3, 5)

            for (ex, ey, ew, eh) in ojos:

                # Se crea un rectángulo en donde se encuentren los ojos.
                cv2.rectangle(
                              reg_color, (ex, ey),
                              (ex + ew, ey + eh), (255, 0, 0), 2
                              )

        # Se muestra la imagen final
        cv2.imshow('Reconocimiento', frame)

        # Se define la tecla para termiar el proceso.
        if cv2.waitKey(1) == ord('q'):
            break

    # Se libera la cámara y se cierran todas las ventanas.
    captura.release()
    cv2.destroyAllWindows()
