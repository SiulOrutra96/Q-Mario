from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import tensorflow as tf

import os
import random
import numpy as np
from PIL import Image
import cv2


# ────────────────────────────────────────────────────────────────────────────────
# Acciones
# ────────────────────────────────────────────────────────────────────────────────

# 0 Detenerse
# 1 Caminar derecha
# 2 Saltar y caminar derecha
# 3 Correr derecha
# 4 Saltar y correr derecha
# 5 Saltar
# 6 Caminar izquierda


def cargarTablaQ():
    tablaQ = {}
    archivo = open(rutaQ, "r")
    
    for linea in archivo:
        estado = ""
        i = 1
        for caracter in linea:
            if caracter == "|":
                break

            estado += caracter
            i += 1
        
        tablaQ[estado] = []
        valor = ""
        for caracter in linea[i:]:
            if caracter == "|":
                tablaQ[estado].append(float(valor))
                valor = ""
            else:
                valor += caracter

    archivo.close()

    return tablaQ

def guardarTablaQ(tablaQ):
    archivo = open(rutaQ, "w")
    for estado in tablaQ:
        archivo.write(estado + "|")
        for valor in tablaQ[estado]:
            archivo.write(str(valor) + "|")
        archivo.write("\n")
    
    archivo.close()

def crearEstado(tablaQ, estado):
    tablaQ[estado] = []
    for i in range(7):
        tablaQ[estado].append(0.5)

def procesarCaptura(captura, posx, posy, index = 0):
    capturaCortada = cortarCaptura(captura, posx, posy)
    capturaReshapeada = np.expand_dims(capturaCortada, axis=0)

    prediccion = modelo.predict([capturaReshapeada])

    resultados = prediccion[0].tolist()
    numeroClase = int(resultados.index(max(resultados)))

    # tuki = input()
    # # mostrar imagen
    # if tuki == "1":
    #     imagen = Image.fromarray(capturaCortada)
    #     imagen.show()

    #     print("Clase: ", CATEGORIAS[numeroClase])
    #     input()

    # guardarCaptura(capturaCortada, index)
    return CATEGORIAS[numeroClase]

def cortarCaptura(captura, posx, posy):
    imagen = Image.fromarray(captura)
    # imagen.show()

    ancho, alto = imagen.size   # Obtener las dimensiones
    left = posx - 20
    if (left < 0):
        left = 0
    elif (left > 191):
        left = 191

    right = left + 64

    top = alto - posy - 5
    if (top < 0):
        top = 0
    elif (top > 175):
        top = 175

    bottom = top + 64

    imagenCortada = imagen.crop((left, top, right, bottom))
    # imagenCortada.show()

    imagenArray = np.array(imagenCortada)

    return imagenArray

def guardarCaptura(captura, index):
    imagen = Image.fromarray(captura)
    imagen.save("capturas/imagen" + str(index) + ".png")
    # imagen.show()

# ────────────────────────────────────────────────────────────────────────────────
# Inicialización de variables
# ────────────────────────────────────────────────────────────────────────────────

tuki = "0"

# Mundo, nivel y modo a jugar
mundo = "1"
nivel = "1"
modo = "0"

# Índice para la toma de capturas
index = 0

# Cantidad de partidas a jugar
partidas = 2000
# Cada cuantas partidas se guarda la tabla Q
guardarCada = 20

# Parámetros para la actualización de la tabla
alfa = 0.4
gama = 0.4

# Factor que determina si se selecciona una acción aleatoria o no
factorAleatoriedad = 5

# Las categorías a las que puede pertenecer una captura
# estos serán los estados en la tabla Q
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

# Ruta para guardar y cargar la tabla Q
rutaQ = "/home/usuario/Documentos/tesina/Q-Mario/Integracion/tablaQ-CNN-Dropout-M" + mundo + "N" + nivel + ".las"

# Ruta del modelo de la red
rutaModelo = "/home/usuario/Documentos/tesina/Q-Mario/CNN/Mario-64X3-Dropout-CNN.model"

# Cargar la red nuronal
modelo = tf.keras.models.load_model(rutaModelo)

# Se inicializa la tabla Q como un hash map
# cada posicion en el hash representa un estado, cada estado contiene un vector de 7 posiciones
# la n-ésima posición del vector contiene la calidad de realizar la n-ésima accion en ese estado
Q = {}

# Si ya existe una tabla Q se carga
if os.path.exists(rutaQ):
    print("***CARGANDO TABLA: tablaQ-CNN-M" + mundo + "N" + nivel +"***")
    Q = cargarTablaQ()
    print("***TABLA Q CARGADA***")

# Imicialización del entorno
env = gym_super_mario_bros.make("SuperMarioBros-" + mundo + "-" + nivel + "-v" + modo)
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

# ────────────────────────────────────────────────────────────────────────────────
# Emtrenamiento
# ────────────────────────────────────────────────────────────────────────────────

for partida in range(partidas):
    print("PARTIDA: ", partida)

    if (not partida == 0) and ((partida % guardarCada == 0) or (partida == partidas - 1)):
        print("***GUARDANDO TABLA: tablaQ-CNN-M" + mundo + "N" + nivel +"***")
        guardarTablaQ(Q)
        print("***TABLA Q GUARDADA***")
    
    # Se reinicia el entrorno cada vez que se empieza una nueva partida
    observacion = env.reset()
    observacion, reward, muerto, info = env.step(0)

    # Se reinicia el estado actual
    estadoActual = procesarCaptura(observacion, info["x_pos"], info["y_pos"])

    # Si el estado actual no se encuentra en la tabla se guarda
    if not estadoActual in Q:
        # TODO - crear estado
        crearEstado(Q, estadoActual)

    # Si mario muere o llega a la bandera la partida termina
    # Se inicializan las variables
    muerto = False
    bandera = False
    puntajeAnterior = 0
    puntajeActual = 0
    posAnterior = 40
    tiempoAnterior = 400
    estatusAnterior = "small"
    vidadAnterior = 2
    recompensa = 0
    paso = -1
    CONSTANTE_ANCLAJE = 90
    puntoAnclaje = CONSTANTE_ANCLAJE
    posXAct = 0
    posXAnt = 40

    while (not muerto) and (not bandera):
        # Esto renderiza el entorno (lo muestra en pantalla)
        env.render()

        # Se determina si la accion será aleatoria o no
        aleatorio = random.randint(0, 100)
        
        # Si el número aleatorio es menor que el factor de aleatoriedad la acción será aleatoria
        if aleatorio <= factorAleatoriedad:
            accionSeleccionada = env.action_space.sample()
        # Si no se selecciona la mejor acción 
        else:
            conjuntoAcciones = Q[estadoActual]
            accionSeleccionada = conjuntoAcciones.index(max(conjuntoAcciones))

        # Se realiza la acción
        observacion, reward, muerto, info = env.step(accionSeleccionada)

        # print("pos: ", info["x_pos"])
        # input()

        # Mover punto de anclaje
        posXAnt = posXAct
        posXAct = info["x_pos"]
        avance = posXAct - posXAnt
        if posXAct > puntoAnclaje and avance > 0:
            puntoAnclaje += avance

        bandera = info["flag_get"]

        if paso == -1 or paso == 4 or bandera or muerto:
            paso = 0

            # Se calcula la recompensa por tomar esta acción
            recompensa = 0
            recompensa = recompensa + (info["score"] - puntajeAnterior)/20
            recompensa = recompensa + (info["time"] - tiempoAnterior)
            recompensa = recompensa + (info["x_pos"] - posAnterior)

            if estatusAnterior == "small" and (info["status"] == "tall" or info["status"] == "fireball"):
                recompensa += 5
            elif estatusAnterior == "tall" and info["status"] == "fireball":
                recompensa += 5
            elif estatusAnterior == "tall" and info["status"] == "small":
                recompensa -= 5
            elif estatusAnterior == "fireball" and (info["status"] == "tall" or info["status"] == "small"):
                recompensa -= 5

            if bandera:
                recompensa += 100
                print("¡¡¡BANDERA!!!")

            if muerto:
                recompensa -= 15
                print("MUERTO :(")
            
            # Se actualizan las variables de la recompensa
            puntajeAnterior = info["score"]
            tiempoAnterior = info["time"]
            estatusAnterior = info["status"]
            posAnterior = info["x_pos"]

            # Se actualiza el estado actual y el anterior
            estadoAnterior = estadoActual
            
            # Se toma captura de pantalla del estado actual y se proces
            # TODO - procesar captura
            parametroX = posXAct - puntoAnclaje + CONSTANTE_ANCLAJE
            estadoActual = procesarCaptura(observacion, parametroX, info["y_pos"], index)
            index += 1 

            # Si el estado actual no se encuentra en la tabla se guarda
            if not estadoActual in Q:
                crearEstado(Q, estadoActual)

            # Se actualiza el valor de la accion seleccionad en la tabla Q con la función de abajo
            # Q(et, at) ← Q(et, at) + α(rt+1 + γ*máxQ(et+1, a) - Q(et, at))

            # Se selecciona el mayor valor de Q en el nuevo estado
            conjuntoAcciones = Q[estadoActual]
            maxQ = max(conjuntoAcciones)

            Q[estadoAnterior][accionSeleccionada] = Q[estadoAnterior][accionSeleccionada] + (alfa * (recompensa + (gama * maxQ) - Q[estadoAnterior][accionSeleccionada]))

        paso += 1
    print("Último X: ", info["x_pos"])
env.close()