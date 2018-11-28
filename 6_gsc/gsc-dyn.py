"""
Implementación del Beamformer GSC utilizando una mu dinámica que cambia con el
tiempo.
"""

import numpy as np
from scipy.io import wavfile
from calcular_distancias import calcular_desfases

##############################################################
########### Cargando audios y separando en ventanas ##########
##############################################################

WINDOW_SIZE = 1024
N_MICS = 3 # Siempre se trabajará con los 3 micrófonos
Nw = 16 # Tamaño del filtro LMS
mu0 = .1 # learning rate base
mu_max = 0.0001 # learning rate maximo

input_directory = "../in/clean-2-source/"
outputfile = "gsc-dyn_ventana{0}_mu01_no-acumulandoG_dir1-exactos.wav".format(Nw)

mics = [] # mics[i] tendrá todo el audio del micrófono #i
for i in range(1, N_MICS + 1):
    sample_rate, mic = wavfile.read(input_directory + "wav_mic" + str(i) + ".wav")
    mics.append(mic)

mic_windows = [] # mic_windows[i][j] tendrá la ventana #j del micrófono #i
for i in range(N_MICS):
    mic_windows.append(np.array_split(mics[i], len(mics[i]) / WINDOW_SIZE))

print("Sample rates:", sample_rate)
print("Length:", len(mics[0]), len(mics[1]), len(mics[2]))
print("Forma del audio de entrada:", mics[0].shape, "Tipo de dato:", mics[0].dtype) # los audios se cargan como int16
print("Ventanas por procesar:", len(mic_windows[0]))

##############################################################
########### Creando matriz de steering vectors ###############
##############################################################

# NOTA: Esta sección se haría antes de comenzar a procesar las ventanas
# de JACK

def make_W(n_mics, frecs, T):
    """
    Función que duelve matriz con steering vectors para hacer desfases

    n_mics: número de micrófonos
    frecs: arreglo con frecuencias a utilizar
    T: arreglo con desfases en segundos con respecto al primer micrófono
    """
    rows = []
    for i in range(n_mics):
        row = np.exp([-1j*2*np.pi*f*T[i] for f in frecs])
        rows.append(row)

    return np.vstack(rows)

DOA = 90 # dirección de interés (ver diapositivas del tema 5 para saber desde dónde se cuenta el ángulo)
DIST = 0.18 # distancia entre micrófonos en metros
VSOUND = 343 # velocidad del sonido
BUFFER_SIZE = WINDOW_SIZE*4 # Para hacer hanneado y overlap and sum es necesario procesar 2 buffers de 4 ventanas cada uno

T = calcular_desfases(DIST, DOA, 'exactos', VSOUND) # Arreglo de desfases de tamaño N_MICS
print()
print("T:", T)

frecs = np.array([(f*sample_rate)/BUFFER_SIZE for f in range(BUFFER_SIZE//2 + 1)]) # frecuencias no negativas
frecs = np.concatenate([frecs, -1*frecs[-2:0:-1]]) # todas las frecuencias (contando las reflejadas)
print("Tamaño de las frecuencias: {0}".format(len(frecs))) # el arreglo de frecuencias tiene tamaño BUFFER_SIZE
print("Frecs", frecs)
print("Max Frec: {0}. Min Frec: {1}.".format(np.max(frecs), np.min(frecs)))

W = make_W(3, frecs, T) # steering vectors; de tamaño N_MICS x BUFFER_SIZE
W_H = np.asarray(np.matrix(W).getH()) # hermitianda de W
print("Tamaño de la hermitiana de W: {0}".format(W_H.shape))
print()

##############################################################
########### Procesando ventanas ##############################
##############################################################

output = np.array([], dtype = np.int16) # aquí se guardaran las ventanas tras procesarse para escribirlas en un archivo
buffers = [] #

for _ in range(N_MICS): # por cada micrófono se tiene un buffer de 6 ventanas:
    buffers.append(np.zeros((WINDOW_SIZE*6), dtype=np.int16))

o_buffer = np.zeros(Nw) # buffer del tamaño del filtro LMS
# g = np.zeros((N_MICS - 1, Nw)) # matriz de filtros # TODO: preguntarle al maestro si se debe matener la matriz de filtros entre iteraciones

for window_index in range(len(mic_windows[0])): # se procesa una ventana a la vez
    if window_index % 250 == 0:
        print("{0} de {1} ventanas".format(window_index, len(mic_windows[0])))

    #########################################################
    ######### DELAY AND SUM #################################
    #########################################################

    # Como es necesario hacer transformada de Fourier, también es necesario
    # hacer Hanning. Para ello es necesario tomar 2 buffers de tamaño
    # BUFFER_SIZE (4 ventanas) hannearlos, y transformarlos, para luego
    # tomar solo una parte de ambos buffers para sumarla. Así se calcula
    # el resultado de una ventana.

    # Debido a lo anterior cuando se reciba la ventana "i", realmente se
    # estaría calculando (y outputeando) la ventana correspondiente a
    # 2.5 ventanas atrás.

    # En general usaré el término "ventana actual" para referirme a la
    # ventana que será el output del tiempo actual.

    # Se guarda la nueva ventana de cada micrófono en su  buffer respectivo
    # desplazando a una ventana vieja en dichos buffers
    for i in range(N_MICS):
        buffers[i] = np.roll(buffers[i], -WINDOW_SIZE)
        buffers[i][-WINDOW_SIZE:] = mic_windows[i][window_index]

    # Por cada micrófono se toma un buffer de 4 ventanas (BUFFER_SIZE)
    # Se hannean y se transforman
    buffer1_fft = []
    for i in range(N_MICS):
        b = buffers[i][:BUFFER_SIZE]*np.hanning(BUFFER_SIZE)
        buffer1_fft.append(np.fft.fft(b))

    # Se toma otro buffer de 4 ventanas.
    # Se hannean y se transforman.
    buffer2_fft = []
    for i in range(N_MICS):
        b = buffers[i][-BUFFER_SIZE:]*np.hanning(BUFFER_SIZE)
        buffer2_fft.append(np.fft.fft(b))

    # Se construyen 2 matrices con cada buffer
    X1 = np.vstack(buffer1_fft) # X1 es una matriz de N_MICS x BUFFER_SIZE
    X2 = np.vstack(buffer2_fft)

    # Se hace el desfase en frecuencia  (delay)
    X1 = X1 * W_H.T
    X2 = X2 * W_H.T

    # Se devuelve al tiempo
    X1 = np.real(np.fft.ifft(X1))
    X2 = np.real(np.fft.ifft(X2))

    # Se suma el buffer respectivo de cada micrófono, tomando solo el área
    # que corresponda a la ventana actual
    HALF_WINDOW_SIZE = int(WINDOW_SIZE / 2)
    X = X1[:,-3*HALF_WINDOW_SIZE:-HALF_WINDOW_SIZE]*.5 + X2[:,HALF_WINDOW_SIZE:3*HALF_WINDOW_SIZE]*.5
    # X es una matriz de tamaño N_MICS x WINDOW_SIZE

    #########################################################
    ######### GSC ###########################################
    #########################################################

    # Para esta sección uso los nombres de variables que usó
    # el maestro en su código de matlab y en sus diapositvias
    # del tema de Beamformers.

    # Resultado del delay and sum:
    y_u = np.sum(X, 0)/N_MICS # arreglo de tamaño WINDOW_SIZE

    # Calculando las x_n y y_n,
    # donde x_n es la diferencia entre señales de entrada
    # adyacentes, y y_n es la señal del ruido estimado
    x_n = np.zeros((N_MICS - 1,WINDOW_SIZE)) # matriz de tamaño N_MICS -1 x WINDOW_SIZE
    for m in range(N_MICS - 1):
        x_n[m] = X[m + 1,:] - X[m,:]
    y_n = sum(x_n, 0) / (N_MICS - 1) # arreglo de tamaño WINDOW_SIZE

    o = np.zeros(WINDOW_SIZE) # arreglo de salida donde se pondrá la ventana actual (de tamaño WINDOW_SIZE)
    o[0:Nw] = o_buffer # Se rellenan los primeros Nw elementos con los últimos Nw elementos del "o" anterior (donde Nw es el tamaño del filtro)

    g = np.zeros((N_MICS - 1, Nw)) # matriz de filtros, de tamaño N_MICS - 1 x Tamaño del filtro
    updater = np.zeros((N_MICS - 1, Nw)) # matriz de actualización de filtros, de tamaño N_MICS - 1 x Tamaño del filtro

    for k in range(Nw, WINDOW_SIZE): # lo siguiente se debe hacer por cada muestra de la ventana actual, excepto por las primeras Nw
        this_y_u = y_u[k]/(2**15) #this_y_u es un escalar; el "2**15" solo se usa en python por cómo se manejan los audios, no es necesario ponerlo en C
        this_x_n = x_n[:, k - Nw : k]/(2**15) # this_x_n es una matriz de tamaño N_MICS - 1 x Tamaño de filtro; de nuevo lo de "2**15" es solo para python
        this_y_n = np.sum(np.sum(g*this_x_n)) # this_y_n es un escalar

        o[k] = this_y_u - this_y_n # se llena la salida "o" una muestra a la vez
        this_o = o[k - Nw:k] # se toman los últimos Nw elementos calculados de o

        # Actualizando el filtro
        p_x_n = np.sum(this_x_n**2, 1) # potencia del ruido; vector del tamaño del filtro
        p_o = np.sum(this_o**2) # potencia de la salida; escalar

        this_mu = mu0*p_x_n/p_o # vector del tamaño del filtro. En la primera iteración p_o = 0, por lo que esto es una división entre 0; python lo ignora pero no sé si C también.

        indices_positivos = this_mu < mu_max # vector booleano de tamaño N_MICS - 1
        indices_negativos = this_mu >= mu_max # vector booleano de tamaño N_MICS - 1

        # A continuación se actualiza la matriz de filtros "g"
        # Recordar que "g" es de tamaño N_MICS - 1 x Tamaño de filtro
        # Y "updater" tiene las mismas dimensiones que "g"
        # Se seleccionan los renglones de upgates que hayan cumplido con la fórmula boolena (indices positivos)
        # y se les actualiza de manera distinta a los otros renglones que no hagan cumplido con ella.

        updater[indices_positivos, :] = mu0*o[k]*this_x_n[indices_positivos,:]/p_o
        updater[indices_negativos, :] = mu0*o[k]*this_x_n[indices_negativos, :] / np.tile(p_x_n[indices_negativos], (Nw, 1)).T
        g = g + updater

    o_buffer = o[-Nw:].copy() # se guardan los últimos Nw elementos de "o" para la siguiente iteración
    output_window = (o*(2**14)).astype(np.int16) # La ventana de salida actual; de nuevo el 2**14 no es necesario en C.
    output = np.concatenate((output, output_window)) # en output se van guardando todas las ventanas procesadas

# Una vez procesadas todas las ventanas se escribe el archivo de audio con todo el audio transformado:
print("Forma del audio de salida:", output.shape, "Tipo de dato:", output.dtype)
wavfile.write(outputfile, sample_rate, output)
print("Listo")
