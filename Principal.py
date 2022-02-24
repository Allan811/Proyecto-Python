#!/usr/bin/python3
"""
Modulo que para el menú y llamado de las funciones en  el
modulo Funciones.
Ejecute solo este archivo.
"""

import Funciones as func


def menu():
    entrada_usuario = int(input('MENU. Ingrese el número de lo que '
                                'desee hacer.\n'
                                '1. Reconocimiento de rostro y ojos.\n'
                                '2. Filtro pizzas.\n'
                                '3. Filtro lazo\n'
                                '4. Cuadro.\n'
                                '5. Cuatro cámaras\n'
                                '6. SALIR\n'
                                '>>> '))
    while True:

        if entrada_usuario == 1:
            print('Pulse "q" para volver al menu principal.\n')
            func.reconocimiento()

        elif entrada_usuario == 2:
            print('Pulse "q" para volver al menu principal.\n')
            func.filtros('Pizzas')

        elif entrada_usuario == 3:
            print('Pulse "q" para volver al menu principal.\n')
            func.filtros('Lazo')

        elif entrada_usuario == 4:
            print('Pulse "q" para volver al menu principal.\n')
            func.cuadro()

        elif entrada_usuario == 5:
            print('Pulse "q" para volver al menu principal.\n')
            func.cuatro_camaras()

        elif entrada_usuario == 6:
            break

        else:
            print('Error: Solo ingrese un número entero del 1 al 6.\n')

        entrada_usuario = int(input('MENU. Ingrese el número de lo que '
                                    'desee hacer.\n'
                                    '1. Reconocimiento de rostro y ojos.\n'
                                    '2. Filtro pizzas.\n'
                                    '3. Filtro lazo\n'
                                    '4. Cuadro.\n'
                                    '5. Cuatro cámaras\n'
                                    '6. SALIR\n'
                                    '>>> '))


menu()
