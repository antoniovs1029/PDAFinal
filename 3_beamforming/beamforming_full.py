"""
Beamforming con Delay and Sum, aplicado sobre las señales completas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

sample_rate, mic1 = wavfile.read("noisy-1-source/wav_mic1.wav")
sample_rate2, mic2 = wavfile.read("noisy-1-source/wav_mic2.wav")
sample_rate3, mic3 = wavfile.read("noisy-1-source/wav_mic3.wav")

print("Sample rates:", sample_rate, sample_rate2, sample_rate3) # deberían ser los mismos
print("Length:", len(mic1), len(mic2), len(mic3)) # deberían ser los mismos
print("Forma del audio de entrada:", mic1.shape, "Tipo de dato:", mic1.dtype)

SIGNAL_SIZE = len(mic1)
#t = list(range(SIGNAL_SIZE))
#plt.plot(t, mic1, t, mic2, t, mic3)
#plt.show()
#plt.close()

def make_W(n_mics, frecs, T):
    rows = []
    for i in range(n_mics):
        row = np.exp([-1j*2*np.pi*f*T[i] for f in frecs])
        rows.append(row)

    return np.vstack(rows)

DOA = 0 # grados
DIST = 0.21 # metros
VSOUND = 343 # metros / s

T = [0, np.sin(np.radians(DOA))*DIST/VSOUND, np.sin(np.radians(DOA - 60))*DIST/VSOUND]
print()
print("T:", T)

frecs = np.array([(f*sample_rate)/SIGNAL_SIZE for f in range(SIGNAL_SIZE//2 + 1)]) # frecuencias no negativas
frecs = np.concatenate([frecs, -1*frecs[-2:0:-1]]) # frecuencias reflejadas
print("Tamaño de las frecuencias: {0}".format(len(frecs)))
print("Frecs", frecs)
print("Max Frec: {0}. Min Frec: {1}.".format(np.max(frecs), np.min(frecs)))

W = make_W(3, frecs, T)
W_H = np.asarray(np.matrix(W).getH())
print("Tamaño de la hermitiana de W: {0}".format(W_H.shape))
print()

mic1_fft = np.fft.fft(mic1)
mic2_fft = np.fft.fft(mic2)
mic3_fft = np.fft.fft(mic3)

X = np.vstack([mic1_fft, mic2_fft, mic3_fft])
S = np.zeros((len(mic1)), dtype=complex)

for f in range(len(mic1)):
    wkk = W_H[f,:]
    xkk = X[:,f]
    multip = np.dot(wkk,xkk)
    S[f] = multip

output = np.real(np.fft.ifft(S)).astype(np.int16)

print("Forma del audio de salida:", output.shape, "Tipo de dato:", output.dtype)

wavfile.write("out_full.wav", sample_rate, output)

print("Listo")
