
"""
TUIA - Procesamiento de Imagenes I
Trabajo Practico N1 - Problema 1: Ecualizacion local de histograma
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ecualizacion_local_histograma(img, tam_ventana):
    """
    Aplica la ecualizacion local de histograma sobre una imagen en escala de grises.
    
    Parametros:
    - img: Imagen de entrada monocromatica (uint8).
    - tam_ventana: Tupla (M, N) con las dimensiones de la ventana de analisis.
    
    Retorna:
    - Imagen ecualizada localmente (uint8).
    """
    # TODO: Implementar logica con padding (cv2.copyMakeBorder) y ventana deslizante
    pass


def main():
    # TODO: Cargar 'Imagen_con_detalles_escondidos.tif'
    # TODO: Ejecutar la ecualizacion local para distintas ventanas y visualizar resultados
    pass


if __name__ == "__main__":
    main()