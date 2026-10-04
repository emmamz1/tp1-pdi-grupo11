
"""
TUIA - Procesamiento de Imagenes I
Trabajo Practico N1 - Problema 1: Ecualizacion local de histograma
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

def ecualizacion_local_histograma(img, tam_ventana):
    """
    Aplica ecualizacion local de histograma sobre una imagen
    en escala de grises. Recibe una imagen uit8 y una tupla con el tamaño de la ventana (M, N)
    y retorna la imagen ecualizada localmente
    """
    M, N = tam_ventana
    alto, ancho = img.shape

    # para ventanas impares
    pad_y = M // 2
    pad_x = N // 2

    img_padded = cv2.copyMakeBorder(img,top=pad_y,bottom=pad_y,left=pad_x,right=pad_x,borderType=cv2.BORDER_REPLICATE)
    img_ecualizada = np.zeros((alto, ancho), dtype=np.uint8)
    total_pixeles_ventana = M * N

    # Recorrer todos los pixeles de la imagen
    for i in range(alto):
        for j in range(ancho):
            ventana = img_padded[i:i + M, j:j + N]
            pixel_central = img[i, j]

            # Histograma local
            hist, _ = np.histogram(
                ventana.flatten(),
                bins=256,
                range=[0, 256]
            )

            cdf = hist.cumsum()
            valores_no_nulos = np.nonzero(cdf)[0] # primero valor no nulo
            cdf_min = cdf[valores_no_nulos[0]]

            # Ecualizacion local
            if total_pixeles_ventana == cdf_min:
                nuevo_valor = pixel_central
            else:
                nuevo_valor = np.round(
                    ((cdf[pixel_central] - cdf_min) /
                     (total_pixeles_ventana - cdf_min)) * 255
                )

            img_ecualizada[i, j] = np.clip(nuevo_valor, 0, 255)

    return img_ecualizada


def main():
    ruta_img = 'Imagen_con_detalles_escondidos.tif'
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: No se encontró la imagen '{ruta_img}' en el directorio actual")
        return

    img_global = cv2.equalizeHist(img)
    ventanas = [(7, 7), (21, 21), (51, 51)]
    resultados_locales = []

    for v in ventanas:
        print(f"Procesando ventana {v[0]}x{v[1]}")
        res = ecualizacion_local_histograma(img, v)
        resultados_locales.append(res)


    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Original")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(img_global, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title("Ecualización Global")
    axs[0, 1].axis('off')
    axs[0, 2].axis('off')  # Espacio libre
    for idx, (v, res) in enumerate(zip(ventanas, resultados_locales)):
        axs[1, idx].imshow(res, cmap='gray', vmin=0, vmax=255)
        axs[1, idx].set_title(f"Local Ventana {v[0]}x{v[1]}")
        axs[1, idx].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()