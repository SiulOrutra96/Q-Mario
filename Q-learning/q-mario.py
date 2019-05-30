from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import os
import random
import numpy as np
from PIL import Image

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
    capturaCortada = achicarCaptura(capturaCortada)
    estado = aplanarCaptura(capturaCortada)
    # guardarCaptura(capturaCortada, index)
    return estado

def cortarCaptura(captura, posx, posy):
    imagen = Image.fromarray(captura)
    # imagen.show()

    ancho, alto = imagen.size   # Obtener las dimensiones
    left = posx - 20
    right = left + 64
    top = alto - posy - 5
    bottom = top + 64
    imagenCortada = imagen.crop((left, top, right, bottom))
    # imagenCortada.show()

    imagenArray = np.array(imagenCortada)

    return imagenArray

def achicarCaptura(captura):
    FACTOR_REDUCCION = 4
    imagen = Image.fromarray(captura)

    ancho, alto = imagen.size
    nuevoAncho = int(ancho/FACTOR_REDUCCION)
    nuevoAlto = int(alto/FACTOR_REDUCCION)

    capturaChica = imagen.resize((nuevoAncho, nuevoAlto), Image.LANCZOS)
    # capturaChica.show()

    imagenArray = np.array(capturaChica)

    return imagenArray

def aplanarCaptura(captura):
    array = np.matrix.flatten(captura)
    cadena = ""
    for pixel in array:
        cadena += str(pixel) + ":"
    
    return cadena

def guardarCaptura(captura, index):
    imagen = Image.fromarray(captura)
    imagen.save("capturas/imagen" + str(index) + ".png")
    # imagen.show()

# ────────────────────────────────────────────────────────────────────────────────
# Inicialización de variables
# ────────────────────────────────────────────────────────────────────────────────

# Ruta para guardar y cargar la tabla Q
rutaQ = "/home/usuario/Documentos/tesina/Q-Mario/Q-learning/tablaQ2.las"

# Se inicializa la tabla Q como un hash map
# cada posicion en el hash representa un estado, cada estado contiene un vector de 7 posiciones
# la n-ésima posición del vector contiene la calidad de realizar la n-ésima accion en ese estado
Q = {}

# Si ya existe una tabla Q se carga
if os.path.exists(rutaQ):
    print("***CARGANDO TABLA Q***")
    Q = cargarTablaQ()
    print("***TABLA Q CARGADA***")

# Cantidad de partidas a jugar
partidas = 2000
# Cada cuantas partidas se guarda la tabla Q
guardarCada = 10

# Parámetros para la actualización de la tabla
alfa = 0.4
gama = 0.4

# Factor que determina si se selecciona una acción aleatoria o no
factorAleatoriedad = 5

# Índice para la toma de capturas
index = 0

# Mundo, nivel y modo a jugar
mundo = "1"
nivel = "1"
modo = "0"

# Imicialización del entorno
env = gym_super_mario_bros.make("SuperMarioBros-" + mundo + "-" + nivel + "-v" + modo)
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

# ────────────────────────────────────────────────────────────────────────────────
# Emtrenamiento
# ────────────────────────────────────────────────────────────────────────────────

for partida in range(partidas):
    print("PARTIDA: ", partida)

    if (not partida == 0) and ((partida % guardarCada == 0) or (partida == partidas - 1)):
        print("***GUARDANDO TABLA Q***")
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
    CONSTANTE_ANCLAJE = 85
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

        # Mover punto de anclaje
        posXAnt = posXAct
        posXAct = info["x_pos"]
        if posXAct > puntoAnclaje:
            avance = posXAct - posXAnt
            puntoAnclaje += avance

        bandera = info["flag_get"]

        if paso == -1 or paso == 5 or bandera or muerto:
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