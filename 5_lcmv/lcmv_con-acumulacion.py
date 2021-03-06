"""
LCMV

Calculando la covarianza con información guardada de cada ventana

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

MIC_NUM = 3 # La implementacion está hecha especificamente para 3 micrófonos, si se cambia, será necesario cambiar todo el código
WINDOW_SIZE = 1024

sample_rate, mic1 = wavfile.read("clean-2-source/wav_mic1.wav")
sample_rate2, mic2 = wavfile.read("clean-2-source/wav_mic2.wav")
sample_rate3, mic3 = wavfile.read("clean-2-source/wav_mic3.wav")

print("Sample rates:", sample_rate, sample_rate2, sample_rate3) # deberían ser los mismos
print("Length:", len(mic1), len(mic2), len(mic3)) # deberían ser los mismos
print("Forma del audio de entrada:", mic1.shape, "Tipo de dato:", mic1.dtype)

t = list(range(len(mic1)))
# plt.plot(t, mic1, t, mic2, t, mic3)
# plt.show()
# plt.close()

mic1_windows = np.array_split(mic1, len(mic1) / WINDOW_SIZE)
mic2_windows = np.array_split(mic2, len(mic2) / WINDOW_SIZE)
mic3_windows = np.array_split(mic3, len(mic3) / WINDOW_SIZE)

output = np.array([], dtype = np.int16)

def make_W(n_mics, frecs, T):
    rows = []
    for i in range(n_mics):
        row = np.exp([-1j*2*np.pi*f*T[i] for f in frecs])
        rows.append(row)

    return np.vstack(rows)

DOA = -30 # grados
DOA_NULL = 90
DIST = 0.18 # metros
VSOUND = 343 # metros / s

T = [0, np.sin(np.radians(DOA))*DIST/VSOUND, np.sin(np.radians(DOA - 60))*DIST/VSOUND]
print()
print("T:", T)

frecs = np.array([(f*sample_rate)/WINDOW_SIZE for f in range(WINDOW_SIZE//2 + 1)]) # frecuencias no negativas
frecs = np.concatenate([frecs, -1*frecs[-2:0:-1]]) # frecuencias reflejadas
print("Tamaño de las frecuencias: {0}".format(len(frecs)))
print("Frecs", frecs)
print("Max Frec: {0}. Min Frec: {1}.".format(np.max(frecs), np.min(frecs)))

W = make_W(3, frecs, T)
print("Tamaño de W: {0}".format(W.shape))
print()

T2 = [0, np.sin(np.radians(DOA_NULL))*DIST/VSOUND, np.sin(np.radians(DOA_NULL - 60))*DIST/VSOUND]
print("T2:", T2)
N = make_W(3, frecs, T2)
print("Tamaño de N: {0}".format(N.shape))

N_SAVED_WINDOWS = 15 # numero de ventanas a guardar
saved_windows = [np.zeros((N_SAVED_WINDOWS, WINDOW_SIZE)) for i in range(MIC_NUM)] # se guardan las ventanas de cada microfono de manera separada


for i in range(len(mic1_windows)):
    if i % 250 == 0:
        print("{0} de {1} ventanas".format(i, len(mic1_windows)))

    window_mic1 = mic1_windows[i]
    window_mic2 = mic2_windows[i]
    window_mic3 = mic3_windows[i]

    window_mic1_fft = np.fft.fft(window_mic1)
    window_mic2_fft = np.fft.fft(window_mic2)
    window_mic3_fft = np.fft.fft(window_mic3)

    X = np.vstack([window_mic1_fft, window_mic2_fft, window_mic3_fft])
    S = np.zeros((WINDOW_SIZE), dtype=complex)

    for m in range(MIC_NUM):
        saved_windows[m] = np.delete(saved_windows[m], 0, 0) # se elimina el primer renglón
        saved_windows[m] = np.vstack([saved_windows[m], X[m]]) # se añade la nueva ventana

    for f in range(WINDOW_SIZE):
        # Se toma la frecuencia f de todas las ventanas guardadas para calcular R:
        prev_windows = []
        for m in range(MIC_NUM):
            prev_windows.append(saved_windows[m][:,f])

        X_F = np.vstack(prev_windows) # la info sobre esta frecuencia

        R = np.dot(X_F, X_F.conj().T)

        for m in range(R.shape[0]):
            R[m][m] = R[m][m]*1.001

        # print(R)
        # input()
        try:
            R_inv = np.linalg.inv(R)
        except:
            print("Error en inversa de R; ventana {0} en frec {0}".format(i, f))
            print(R)

        C = np.column_stack([W[:,f], N[:,f]])
        a1 = np.dot(R_inv, C)
        a2 =  np.dot(R_inv, C)
        a3 = np.dot(w_a.conj().T, np.dot(R_inv, C)) # TODO: De dónde salió esta w_a???
        for m in range(a3.shape[0]):
            a3[m][m] = 1.001*a3[m][m]

        try:
            a4 = np.linalg.inv(a3)
        except:
            print("Error en inversa de a3; ventana {0} en frec {0}".format(i, f))
        
        a = np.dot(a1, a4)
        """
        print("W_A:")
        print(w_a)
        print("W_A.conj.T:")
        print(w_a.conj().T)
        print("R:")
        print(R)
        print("R.inv:")
        print(R_inv)
        print("a1\n", a1)
        print("a2\n", a2)
        print("a3\n", a3)
        print("a4\n", a4)
        print("a\n", a)
        input()
        """
        S[f] = np.dot(X[:,f],a[:,0])

    output_window = np.real(np.fft.ifft(S)).astype(np.int16)

    output = np.concatenate((output, output_window))

print("Forma del audio de salida:", output.shape, "Tipo de dato:", output.dtype)

wavfile.write("out_lcmv_con-acumulacion.wav", sample_rate, output)

print("Listo")
