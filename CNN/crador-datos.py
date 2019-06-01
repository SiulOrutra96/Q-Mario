import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

TAMANIO_IMG = 64
CARPETA_IMAGENES = "/home/usuario/Documentos/tesina/Q-Mario/capturas"
CATEGORIAS = ["cubo_arriba",
              "cubo_derecha",
              "cubo_izquierda",
              "enemigo_abajo",
              "enemigo_derecha",
              "hongo_derecha",
              "no_peligro",
              "obstaculo_derecha",
              "obstaculo_izquierda",
              "precipicio_chico_derecha",
              "precipicio_grande_derecha",
              "precipicio_izquierda"]

datosEntrenamiento = []

def crearDatosEntrenamiento():
    for categoria in CATEGORIAS:
        ruta = os.path.join(CARPETA_IMAGENES, categoria)  # crea la ruta a la carpeta de cada categoría
        numeroClase = CATEGORIAS.index(categoria)  # convierte la clase en un número entero

        for img in os.listdir(ruta):  # itera en cada imagen de cada categoría
            imgArray = cv2.imread(os.path.join(ruta, img))  # convierte a arreglo
            datosEntrenamiento.append([imgArray, numeroClase])
            
            # mostrar la imagen
            # cv2.imshow('imagen', img_array)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

crearDatosEntrenamiento()

# desordenar los datos
random.shuffle(datosEntrenamiento)

X = []  # imagenes
y = []  # etiquetas

for imagen, etiqueta in datosEntrenamiento:
    X.append(imagen)
    y.append(etiqueta)

X = np.array(X).reshape(-1, TAMANIO_IMG, TAMANIO_IMG, 3)

# guardar las imagenes
pickle_out = open("/home/usuario/Documentos/tesina/Q-Mario/CNN/X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

# guardar las etiquetas
pickle_out = open("/home/usuario/Documentos/tesina/Q-Mario/CNN/y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()