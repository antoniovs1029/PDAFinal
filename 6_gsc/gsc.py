import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

##############################################################
########### Cargando audios y separando en ventanas ##########
##############################################################

WINDOW_SIZE = 1024
N_MICS = 3 # Si se cambia, se debera cambiar el resto del código
mu = .03
Nw = 100 # Tamaño del filtro
directory = "../in/clean-2-source/"
outputfile = "out_hanneado-noisy2.wav"
mics = []

for i in range(1, N_MICS + 1):
    sample_rate, mic = wavfile.read(directory + "wav_mic" + str(i) + ".wav")
    mics.append(mic)

mic_windows = []
for i in range(N_MICS):
    mic_windows.append(np.array_split(mics[i], len(mics[i]) / WINDOW_SIZE))

print("Sample rates:", sample_rate) # deberían ser los mismos
print("Length:", len(mics[0]), len(mics[1]), len(mics[2])) # deberían ser los mismos
print("Forma del audio de entrada:", mics[0].shape, "Tipo de dato:", mics[0].dtype)
print("Ventanas por procesar:", len(mic_windows[0]))

##############################################################
########### Creando matriz de steering vectors ###############
##############################################################

def make_W(n_mics, frecs, T):
    rows = []
    for i in range(n_mics):
        row = np.exp([-1j*2*np.pi*f*T[i] for f in frecs])
        rows.append(row)

    return np.vstack(rows)

DOA = 0 # grados
DIST = 0.18 # metros
VSOUND = 343 # metros / s
BUFFER_SIZE = WINDOW_SIZE*4 # Para hacer hanneado y overlap and sum es necesario procesar un buffer de 4 ventanas

T = [0, -np.sin(np.radians(DOA))*DIST/VSOUND, -np.sin(np.radians(DOA - 60))*DIST/VSOUND]
print()
print("T:", T)

frecs = np.array([(f*sample_rate)/BUFFER_SIZE for f in range(BUFFER_SIZE//2 + 1)]) # frecuencias no negativas
frecs = np.concatenate([frecs, -1*frecs[-2:0:-1]]) # frecuencias reflejadas
print("Tamaño de las frecuencias: {0}".format(len(frecs)))
print("Frecs", frecs)
print("Max Frec: {0}. Min Frec: {1}.".format(np.max(frecs), np.min(frecs)))

W = make_W(3, frecs, T)
W_H = np.asarray(np.matrix(W).getH())
print("Tamaño de la hermitiana de W: {0}".format(W_H.shape))
print()

##############################################################
########### Procesando ventanas ##############################
##############################################################

output = np.array([], dtype = np.int16)
buffers = []

for _ in range(N_MICS):
    buffers.append(np.zeros((WINDOW_SIZE*6), dtype=np.int16))

o_buffer = np.zeros(Nw)
#for window_index in range(251):
for window_index in range(len(mic_windows[0])):
    if window_index % 250 == 0:
        print("{0} de {1} ventanas".format(window_index, len(mic_windows[0])))

    #########################################################
    ######### DELAY AND SUM #################################
    #########################################################

    # Se guarda la nueva ventana desplazando a una vieja
    for i in range(N_MICS):
        buffers[i] = np.roll(buffers[i], -WINDOW_SIZE)
        buffers[i][-WINDOW_SIZE:] = mic_windows[i][window_index]

    buffer1_fft = []
    for i in range(N_MICS):
        b = buffers[i][:BUFFER_SIZE]*np.hanning(BUFFER_SIZE)
        buffer1_fft.append(np.fft.fft(b))

    buffer2_fft = []
    for i in range(N_MICS):
        b = buffers[i][-BUFFER_SIZE:]*np.hanning(BUFFER_SIZE)
        buffer2_fft.append(np.fft.fft(b))

    X1 = np.vstack(buffer1_fft)
    X2 = np.vstack(buffer2_fft)

    X1 = X1 * W_H.T
    X2 = X2 * W_H.T

    X1 = np.real(np.fft.ifft(X1))
    X2 = np.real(np.fft.ifft(X2))

    HALF_WINDOW_SIZE = int(WINDOW_SIZE / 2)
    X = X1[:,-3*HALF_WINDOW_SIZE:-HALF_WINDOW_SIZE]*.5 + X2[:,HALF_WINDOW_SIZE:3*HALF_WINDOW_SIZE]*.5

    #########################################################
    ######### GSC ###########################################
    #########################################################

    # Resultado del delay and sum:
    y_u = np.sum(X, 0)

    # Calculando las x_n y y_n:
    x_n = np.zeros((N_MICS - 1,WINDOW_SIZE))
    for m in range(N_MICS - 1):
        x_n[m] = X[m + 1,:] - X[m,:]
    y_n = sum(x_n, 0) / (N_MICS - 1)

    # Aplicando beamformer
    o = np.zeros(WINDOW_SIZE)
    o[0:Nw] = o_buffer
    g = np.zeros((N_MICS - 1, Nw))

    for k in range(Nw, WINDOW_SIZE):
        this_y_u = y_u[k]/(2**15)
        this_x_n = x_n[:, k - Nw : k]/(2**15)
        this_y_n = np.sum(np.sum(g*this_x_n))

        o[k] = this_y_u - this_y_n
        this_o = o[k - Nw:k]

        g = g + mu*(np.tile(this_o, (N_MICS - 1 , 1)))*this_x_n

    #print(this_x_n, '\n', o, '\n', g)
    #input()

    o_buffer = o[-Nw:].copy()
    output_window = (o*(2**14)).astype(np.int16)
    output = np.concatenate((output, output_window))

print("Forma del audio de salida:", output.shape, "Tipo de dato:", output.dtype)

wavfile.write(outputfile, sample_rate, output)

print("Listo")
