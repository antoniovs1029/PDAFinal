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

WINDOW_SIZE = 1024
N_MICS = 3 # Si se cambia, se debera cambiar el resto del código
directory = "../in/clean-2-source/"
outfile = "../out/1_con-retrasos-geometricos/out_hanneado-clean-90.wav"
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

DOA = 90 # grados
DIST = 0.18 # metros
VSOUND = 343 # metros / s
BUFFER_SIZE = WINDOW_SIZE*4 # Para hacer hanneado y overlap and sum es necesario procesar un buffer de 4 ventanas

def distancia_a_recta(x, y, pendiente):
    num = -pendiente*x + y
    denom = np.sqrt(pendiente**2 + 1)
    return num/denom

def calcular_desfases(distancia, angulo, vsound):
    radio = (distancia*np.sqrt(3))/3
    cateto_y = np.sqrt(  radio**2 - (distancia/2)**2 )

    x1, y1 = distancia/2, cateto_y
    x2, y2 = -distancia/2, cateto_y
    x3, y3 = 0, -radio

    pendiente = np.tan(np.radians(-angulo + 90)) # Se debe poner asi pues las formulas son usando el angulo del eje x en sentido antihorario, pero el angulo de aire es desde el Y en sentido horario.

    if pendiente == 0:
        d1 = x1/vsound
        d2 = x2/vsound
        d3 = x3/vsound
    else:
        pendiente_inversa = -1 / pendiente
        d1 = distancia_a_recta(x1, y1, pendiente_inversa)/vsound
        d2 = distancia_a_recta(x2, y2, pendiente_inversa)/vsound
        d3 = distancia_a_recta(x3, y3, pendiente_inversa)/vsound

    return [d1 - d1, d2 - d1, d3 - d1]

T = calcular_desfases(DIST, DOA, VSOUND)
T_con_formula = [0, np.sin(np.radians(DOA))*DIST/VSOUND, np.sin(np.radians(DOA - 60))*DIST/VSOUND]
print()
print("T_con_formula:", T_con_formula)
print("T:", T)

frecs = np.array([(f*sample_rate)/BUFFER_SIZE for f in range(BUFFER_SIZE//2 + 1)]) # frecuencias no negativas
frecs = np.concatenate([frecs, -1*frecs[-2:0:-1]]) # frecuencias reflejadas
print("Tamaño de las frecuencias: {0}".format(len(frecs)))
print("Frecs", frecs)
print("Max Frec: {0}. Min Frec: {1}.".format(np.max(frecs), np.min(frecs)))

def make_W(frecs, T):
    rows = []
    for i in range(len(T)):
        row = np.exp([-1j*2*np.pi*f*T[i] for f in frecs])
        rows.append(row)

    return np.vstack(rows)

W = make_W(frecs, T)
W_H = np.asarray(np.matrix(W).getH())
print("Tamaño de la hermitiana de W: {0}".format(W_H.shape))
print()

##############################################################
########### Procesando ventanas ##############################
##############################################################

output = np.array([], dtype = np.int16)
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

        w_a = W[:,f]
        a1 = np.dot(R1_inv, w_a)/np.dot(w_a.conj(), np.dot(R1_inv, w_a))
        a2 = np.dot(R2_inv, w_a)/np.dot(w_a.conj(), np.dot(R2_inv, w_a))

        S1[f] = np.dot(X1[:,f],a1.conj())
        S2[f] = np.dot(X2[:,f],a2.conj())

#        S1[f] = np.dot(X1[:,f],a1)
#        S2[f] = np.dot(X2[:,f],a2)

    output_buffer1 = np.real(np.fft.ifft(S1)).astype(np.int16)
    output_buffer2 = np.real(np.fft.ifft(S2)).astype(np.int16)

    HALF_WINDOW_SIZE = int(WINDOW_SIZE / 2)
    output_window = output_buffer1[-3*HALF_WINDOW_SIZE:-HALF_WINDOW_SIZE] + output_buffer2[HALF_WINDOW_SIZE:3*HALF_WINDOW_SIZE]
    output = np.concatenate((output, output_window))

maxx = np.max(output)
coef = 30000 / maxx
output = output*coef
output = output.astype(dtype=np.int16)
print("Forma del audio de salida:", output.shape, "Tipo de dato:", output.dtype)

wavfile.write(outfile, sample_rate, output)

print("Listo")
