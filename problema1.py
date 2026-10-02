
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
    # Obtener las dimensiones de la ventana y de la imagen
    M, N = tam_ventana
    alto, ancho = img.shape
    
    # Calcular el margen necesario en cada lado para centrar la ventana
    pad_y = M // 2
    pad_x = N // 2
    
    # Agregar borde a la imagen para que la ventana pueda posicionarse en los extremos
    img_padded = cv2.copyMakeBorder(
        img, 
        top=pad_y, 
        bottom=pad_y, 
        left=pad_x, 
        right=pad_x, 
        borderType=cv2.BORDER_REPLICATE
    )
    # Matriz vacia para guardar el resultado pixel por pixel
    img_ecualizada = np.zeros((alto, ancho), dtype=np.uint8)
    total_pixeles_ventana = M * N

    #completar el codigo para recorrer cada píxel de la imagen original
    pass


def main():
    ruta_img = 'Imagen_con_detalles_escondidos.tif'
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        print(f"Error: No se encontró la imagen '{ruta_img}' en el directorio actual.")
        return

    # Definir las ventanas de análisis
    ventanas = [(7, 7), (21, 21), (51, 51)]
    resultados_locales = []

   #completar el codigo para aplicar la ecualizacion local de histograma con cada ventana y guardar los resultados
    pass


if __name__ == "__main__":
    main()