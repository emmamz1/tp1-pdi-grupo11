# PROBLEMA 2
"""
TUIA - Procesamiento de Imagenes I
Trabajo Practico N1 - Problema 2: Validacion de planilla de calificaciones

Uso:
    python problema2.py                      # procesa grade_sheet_*.png (excepto _empty)
    python problema2.py planilla1.png ...    # procesa las imagenes indicadas
    python problema2.py --mostrar            # ademas muestra los pasos intermedios
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Parametros -------------------------------------------------------------
TH_GRIS = 128            # umbral de binarizacion: pixel "tinta" si img < TH_GRIS
TH_FILA = 0.5            # fraccion del ancho de la imagen para considerar linea horizontal
TH_COL = 0.8             # fraccion del alto de la tabla para considerar linea vertical
TH_AREA = 1              # area minima de una componente (descarta pixeles sueltos)
TH_ESPACIO = 0.5         # gap > TH_ESPACIO * altura de caracter => separacion entre palabras
N_REGISTROS = 20

CAMPOS = ["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
CAMPOS_CSV = ["ID", "Legajo", "Nombre y Apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
DIR_SALIDA = "resultados"


# --- Deteccion de la grilla -------------------------------------------------
def detectar_lineas(mascara):
    """Devuelve [(inicio, fin), ...] de cada tramo consecutivo en True (lineas de >1 px)."""
    idx = np.flatnonzero(mascara)
    if len(idx) == 0:
        return []
    tramos = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(int(t[0]), int(t[-1])) for t in tramos]


def detectar_grilla(img_th):
    """
    Detecta las lineas horizontales y verticales de la tabla mediante la suma
    de pixeles por fila y por columna. Devuelve (filas, columnas) como listas
    de (inicio, fin) de cada linea, considerando solo las 20 filas de registros.
    """
    h, w = img_th.shape
    img_rows = np.sum(img_th, 1)
    filas = detectar_lineas(img_rows > TH_FILA * w)
    # Las ultimas 21 lineas horizontales delimitan los 20 registros
    filas = filas[-(N_REGISTROS + 1):]

    y0, y1 = filas[0][0], filas[-1][1]
    img_cols = np.sum(img_th[y0:y1 + 1, :], 0)
    columnas = detectar_lineas(img_cols > TH_COL * (y1 - y0))
    return filas, columnas


# --- Analisis de caracteres -------------------------------------------------
def obtener_caracteres(celda_th):
    """
    Obtiene las cajas (x, y, w, h) de los caracteres de una celda binaria,
    ordenadas de izquierda a derecha. Las componentes que se superponen en x
    (ej.: la tilde de la Ñ) se unen en un solo caracter.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(celda_th, 8, cv2.CV_32S)
    stats = stats[1:]                              # descarta el fondo
    ix_area = stats[:, -1] > TH_AREA
    stats = stats[ix_area, :]
    stats = stats[np.argsort(stats[:, 0])]

    cajas = []
    for x, y, w, h, _ in stats:
        if cajas and x <= cajas[-1][0] + cajas[-1][2] - 1:
            # Se superpone horizontalmente con el caracter anterior: unir
            cx, cy, cw, ch = cajas[-1]
            nx, ny = min(cx, x), min(cy, y)
            cajas[-1] = [nx, ny, max(cx + cw, x + w) - nx, max(cy + ch, y + h) - ny]
        else:
            cajas.append([int(x), int(y), int(w), int(h)])
    return cajas


def contar_palabras(cajas):
    """Cuenta palabras: un gap entre caracteres mayor a TH_ESPACIO*altura es un espacio."""
    if not cajas:
        return 0
    alto = max(c[3] for c in cajas)
    palabras = 1
    for ant, sig in zip(cajas[:-1], cajas[1:]):
        gap = sig[0] - (ant[0] + ant[2])
        if gap > TH_ESPACIO * alto:
            palabras += 1
    return palabras


# --- Validacion de campos ---------------------------------------------------
# Los espacios no cuentan como caracteres (no estan en el conjunto de caracteres
# permitidos); solo separan palabras.
def validar_legajo(n_car, n_pal):
    return n_car == 8 and n_pal == 1


def validar_nombre(n_car, n_pal):
    return n_pal >= 2 and n_car <= 12


def validar_nota(n_car, n_pal):
    return 1 <= n_car <= 2 and n_pal == 1


def validar_condicion(n_car, n_pal):
    return n_car == 1


VALIDADORES = [validar_legajo, validar_nombre, validar_nota, validar_nota, validar_nota, validar_condicion]


def clasificar_condicion(caracter_th):
    """
    Distingue A, L y R a partir de la forma del caracter:
      - L no tiene agujeros.
      - R tiene el trazo vertical izquierdo completo; A no.
    """
    contornos, jerarquia = cv2.findContours(caracter_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    agujeros = 0 if jerarquia is None else int(np.sum(jerarquia[0][:, 3] >= 0))
    if agujeros == 0:
        return "L"
    # Fraccion de pixeles con tinta en la columna izquierda de la caja
    columna_izq = caracter_th[:, :2].max(axis=1) > 0
    return "R" if columna_izq.mean() > 0.8 else "A"


# --- Procesamiento de una planilla -----------------------------------------
def procesar_planilla(ruta_imagen, mostrar=False):
    """
    Procesa y valida los campos de una planilla de calificaciones.
    Devuelve una lista de registros (dict) con el resultado de cada campo,
    el crop del nombre y la condicion detectada.
    """
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(ruta_imagen)
    img_th = img < TH_GRIS
    img_th_ones = img_th.astype(np.uint8)

    filas, columnas = detectar_grilla(img_th_ones)
    if len(filas) != N_REGISTROS + 1 or len(columnas) != len(CAMPOS) + 2:
        raise ValueError(f"{ruta_imagen}: grilla no detectada ({len(filas)} filas, {len(columnas)} columnas)")

    if mostrar:
        mostrar_pasos(img, img_th_ones, filas, columnas)

    registros = []
    for r in range(N_REGISTROS):
        ya, yb = filas[r][1] + 1, filas[r + 1][0]           # interior de la fila
        resultados, crop_nombre, condicion = [], None, None
        # La columna 0 es "Nro.", los campos van de la 1 a la 6
        for c, validar in enumerate(VALIDADORES, start=1):
            xa, xb = columnas[c][1] + 1, columnas[c + 1][0]
            celda_th = img_th_ones[ya:yb, xa:xb]
            cajas = obtener_caracteres(celda_th)
            ok = validar(len(cajas), contar_palabras(cajas))
            resultados.append("OK" if ok else "MAL")

            if c == 2:
                crop_nombre = img[ya:yb, xa:xb]
            if c == 6 and ok:
                x, y, w, h = cajas[0]
                condicion = clasificar_condicion(celda_th[y:y + h, x:x + w] * 255)
        registros.append({"id": r + 1, "resultados": resultados,
                          "crop_nombre": crop_nombre, "condicion": condicion})
    return registros


def mostrar_pasos(img, img_th, filas, columnas):
    """Visualiza los pasos intermedios de la deteccion de la grilla."""
    img_rows = np.sum(img_th, 1)
    y0, y1 = filas[0][0], filas[-1][1]
    img_cols = np.sum(img_th[y0:y1 + 1, :], 0)

    grilla = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for a, b in filas:
        grilla[a:b + 1, :] = (0, 0, 255)
    for a, b in columnas:
        grilla[y0:y1 + 1, a:b + 1] = (255, 0, 0)

    plt.figure(figsize=(14, 8))
    ax = plt.subplot(2, 2, 1); plt.imshow(img_th, cmap="gray"); plt.title("Imagen umbralizada (img < th)")
    plt.subplot(2, 2, 2); plt.plot(img_rows); plt.axhline(TH_FILA * img.shape[1], color="r")
    plt.title("Suma por filas"); plt.xlabel("fila")
    plt.subplot(2, 2, 3); plt.plot(img_cols); plt.axhline(TH_COL * (y1 - y0), color="r")
    plt.title("Suma por columnas (zona de registros)"); plt.xlabel("columna")
    plt.subplot(2, 2, 4); plt.imshow(grilla[..., ::-1]); plt.title("Lineas detectadas")
    plt.tight_layout()
    plt.show()


# --- Salidas ----------------------------------------------------------------
def imprimir_reporte(registros):
    for reg in registros:
        print(f"> Registro {reg['id']}:")
        for campo, res in zip(CAMPOS, reg["resultados"]):
            print(f"> {campo}: {res}")
        print(">")


def guardar_csv(registros, ruta_csv):
    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CAMPOS_CSV)
        for reg in registros:
            writer.writerow([reg["id"]] + reg["resultados"])


def generar_imagen_desaprobados(registros, ruta_salida):
    """
    Genera una unica imagen con el crop del Nombre y Apellido de los alumnos
    correctamente cargados con Condicion Final "R" (recupera) o "L" (libre).
    """
    colores = {"R": (0, 140, 255), "L": (0, 0, 220)}       # BGR: naranja / rojo
    filas = []
    for reg in registros:
        if all(r == "OK" for r in reg["resultados"]) and reg["condicion"] in colores:
            crop = cv2.cvtColor(reg["crop_nombre"], cv2.COLOR_GRAY2BGR)
            crop = cv2.copyMakeBorder(crop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=colores[reg["condicion"]])
            # Indicador: la letra de la Condicion Final ("R" o "L") en el color del recuadro
            etiqueta = np.full((crop.shape[0], 40, 3), 255, np.uint8)
            cv2.putText(etiqueta, reg["condicion"], (10, crop.shape[0] // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colores[reg["condicion"]], 2)
            filas.append(np.hstack([crop, etiqueta]))

    if filas:
        ancho = max(f.shape[1] for f in filas)
        filas = [cv2.copyMakeBorder(f, 2, 2, 0, ancho - f.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for f in filas]
        salida = np.vstack(filas)
    else:
        salida = np.full((40, 400, 3), 255, np.uint8)
        cv2.putText(salida, "Sin alumnos desaprobados", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(ruta_salida, salida)
    return salida


def main():
    args = sys.argv[1:]
    mostrar = "--mostrar" in args
    rutas = [a for a in args if a != "--mostrar"]
    if not rutas:
        rutas = sorted(r for r in glob.glob("grade_sheet_*.png") if "empty" not in r)
    if not rutas:
        print("No se encontraron planillas grade_sheet_<id>.png en el directorio actual.")
        return

    os.makedirs(DIR_SALIDA, exist_ok=True)
    # Procesamiento ciclico de todas las planillas
    for ruta in rutas:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        print(f"\n===== {nombre} =====")
        registros = procesar_planilla(ruta, mostrar)
        imprimir_reporte(registros)

        ruta_csv = os.path.join(DIR_SALIDA, f"{nombre}_validacion.csv")
        ruta_img = os.path.join(DIR_SALIDA, f"{nombre}_desaprobados.png")
        guardar_csv(registros, ruta_csv)
        salida = generar_imagen_desaprobados(registros, ruta_img)
        print(f"CSV guardado en: {ruta_csv}")
        print(f"Imagen de desaprobados guardada en: {ruta_img}")

        if mostrar:
            plt.figure(figsize=(6, 6))
            plt.imshow(salida[..., ::-1]); plt.title(f"{nombre}: alumnos que no aprobaron"); plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
