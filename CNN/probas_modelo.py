import tensorflow as tf
import cv2
import numpy as np

rutaModelo = "/home/usuario/Documentos/tesina/Q-Mario/CNN/Mario-64X3-CNN.model"
modelo = tf.keras.models.load_model(rutaModelo)

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

def cargarImage(ruta):
    imgArray = cv2.imread(ruta)
    nuevoArray = np.expand_dims(imgArray, axis=0)
    # nuevoArray = cv2.resize(imgArray, (64, 64))
    # imgArray.reshape(-1, 64, 64, 3)

    # print("Tukiki: ", nuevoArray.shape)
    # input()

    # mostrar la imagen
    cv2.imshow('imagen', imgArray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return nuevoArray

for i in range(20):
    imagen = cargarImage("/home/usuario/Documentos/tesina/Q-Mario/otras capturas/imagen" + str(15453 + i*100) + ".png")

    prediccion = modelo.predict([imagen])

    resultados = prediccion[0].tolist()
    numeroClase = resultados.index(max(resultados))
    print(CATEGORIAS[int(numeroClase)])
