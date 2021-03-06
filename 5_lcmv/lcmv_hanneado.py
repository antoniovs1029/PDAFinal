"""
Beamforming con Delay and Sum

Calculando es desplazamiento frecuencia a frecuencia en lugar de multiplicar con matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

##############################################################
########### Cargando audios y separando en ventanas ##########
##############################################################

outputfile = "out_hanneado-clean-no_conj1.wav"

WINDOW_SIZE = 1024
N_MICS = 3 # Si se cambia, se debera cambiar el resto del código
directory = "clean-2-source/"
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

def make_W(frecs, T):
    rows = []
    for i in range(len(T)):
        row = np.exp([-1j*2*np.pi*f*T[i] for f in frecs])
        rows.append(row)

    return np.vstack(rows)

DOA = -30 # grados
DOA_NULL = 90
DIST = 0.18 # metros
VSOUND = 343 # metros / s
BUFFER_SIZE = WINDOW_SIZE*4 # Para hacer hanneado y overlap and sum es necesario procesar un buffer de 4 ventanas

T = [0, np.sin(np.radians(DOA))*DIST/VSOUND, np.sin(np.radians(DOA - 60))*DIST/VSOUND]
print()
print("T:", T)

frecs = np.array([(f*sample_rate)/BUFFER_SIZE for f in range(BUFFER_SIZE//2 + 1)]) # frecuencias no negativas
frecs = np.concatenate([frecs, -1*frecs[-2:0:-1]]) # frecuencias reflejadas
print("Tamaño de las frecuencias: {0}".format(len(frecs)))
print("Frecs", frecs)
print("Max Frec: {0}. Min Frec: {1}.".format(np.max(frecs), np.min(frecs)))

W = make_W(frecs, T)
print("Tamaño de W: {0}".format(W.shape))
print()

T2 = [0, np.sin(np.radians(DOA_NULL))*DIST/VSOUND, np.sin(np.radians(DOA_NULL - 60))*DIST/VSOUND]
print("T2:", T2)
N = make_W(frecs, T2)
print("Tamaño de N: {0}".format(N.shape))

##############################################################
########### Procesando ventanas ##############################
##############################################################

def calc_a_arr(W_F, N_F, R_inv):
    C = np.column_stack([W_F, N_F])
    num = np.dot(R_inv, C)
    denom = np.dot(C.conj().T, np.dot(R_inv, C))
    for m in range(denom.shape[0]):
        denom[m][m] = 1.001*denom[m][m] + 0.001

    try:
        denom = np.linalg.inv(denom)
    except:
        print("Error en inversa del denom de A_arr")
    
    return np.dot(num, denom)

# output = np.array([], dtype = np.int16)
output = np.array([])
buffers = []

N_SAVED_WINDOWS = 15 # numero de ventanas a guardar para el calculo de las R's
saved_windows1 = [np.zeros((N_SAVED_WINDOWS, BUFFER_SIZE)) for _ in range(N_MICS)] # se guardan las ventanas de cada microfono de manera separada
saved_windows2 = [np.zeros((N_SAVED_WINDOWS, BUFFER_SIZE)) for _ in range(N_MICS)] # un arreglo para cada R

for _ in range(N_MICS): #inicializando un buffer de 6*WINDOW_SIZE para cada microfono
    buffers.append(np.zeros((WINDOW_SIZE*6), dtype=np.int16))

for window_index in range(251):
# for window_index in range(len(mic_windows[0])):
    if window_index % 50 == 0:
        print("{0} de {1} ventanas".format(window_index, len(mic_windows[0])))

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
    S1 = np.zeros((BUFFER_SIZE), dtype=complex)
    S2 = np.zeros((BUFFER_SIZE), dtype=complex)

    for m in range(N_MICS): # Para luego calcular la R
        saved_windows1[m] = np.delete(saved_windows1[m], 0, 0)
        saved_windows1[m] = np.vstack([saved_windows1[m], X1[m]])

        saved_windows2[m] = np.delete(saved_windows2[m], 0, 0)
        saved_windows2[m] = np.vstack([saved_windows2[m], X2[m]])

    for f in range(BUFFER_SIZE):
        # Se toma la frecuencia f de todas las ventanas guardadas:
        prev_windows1 = []
        prev_windows2 = []
        for m in range(N_MICS):
            prev_windows1.append(saved_windows1[m][:,f])
            prev_windows2.append(saved_windows2[m][:,f])

        X_F1 = np.vstack(prev_windows1)
        X_F2 = np.vstack(prev_windows2)

        R1 = np.dot(X_F1, X_F1.conj().T)
        R2 = np.dot(X_F2, X_F2.conj().T)

        for m in range(R1.shape[0]):
            R1[m][m] = 1.001*R1[m][m] + .001
            R2[m][m] = 1.001*R2[m][m] + .001

        try:
            R1_inv = np.linalg.inv(R1)
        except:
            print("Error en ventana #{0} en la frec {1} en la inversa de R1".format(window_index,f))
            R1_inv = R1

        try:
            R2_inv = np.linalg.inv(R2)
        except:
            print("Error en ventana #{0} en la frec {1} en la inversa de R2".format(window_index,f))
            R2_inv = R2

        a_arr1 = calc_a_arr(W[:,f], N[:,f], R1_inv)
        a_arr2 = calc_a_arr(W[:,f], N[:,f], R2_inv)

        S1[f] = np.dot(X1[:,f],a_arr1[:,0])
        S2[f] = np.dot(X2[:,f],a_arr2[:,0])

#        S1[f] = np.dot(X1[:,f],a_arr1[:,0].conj())
#        S2[f] = np.dot(X2[:,f],a_arr2[:,0].conj())

    output_buffer1 = np.real(np.fft.ifft(S1))
    output_buffer2 = np.real(np.fft.ifft(S2))

    # print(window_index, np.max(output_buffer1), np.max(output_buffer2))

    HALF_WINDOW_SIZE = int(WINDOW_SIZE / 2)
    output_window = output_buffer1[-3*HALF_WINDOW_SIZE:-HALF_WINDOW_SIZE] + output_buffer2[HALF_WINDOW_SIZE:3*HALF_WINDOW_SIZE] 

    output = np.concatenate((output, output_window))

print("Forma del audio de salida:", output.shape, "Tipo de dato (sin normalizar):", output.dtype)
# t = list(range(len(output)))
# plt.plot(output)
# plt.savefig("antes-de-normalizar.png")

maxx = np.max(output[WINDOW_SIZE*10:])
coef = 30000 / maxx
output = output*coef
output = output.astype(dtype=np.int16)

print("Forma del audio de salida:", output.shape, "Tipo de dato (normalizado):", output.dtype)
# t = list(range(len(output)))
# plt.plot(output)
# plt.savefig("tras-normalizar.png")

wavfile.write(outputfile, sample_rate, output)

print("Listo")
