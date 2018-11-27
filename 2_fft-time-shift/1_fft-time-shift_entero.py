# Funciona cuando el desfase es un numero entero del numero de muestras.
# basado en http://qingkaikong.blogspot.com/2016/03/shift-signal-in-frequency-domain.html

import numpy as np
import matplotlib.pyplot as plt

SAMPLING_RATE = 128.0 # 128 puntos por segundo. En general *parece* necesario que la señal a shiftear tenga como tamaño a una potencia de 2
t = np.arange(0, 1, 1/SAMPLING_RATE) # eje del tiempo, durará 1 segundo

FREQ1 = 5
FREQ2 = 2.5

signal = np.sin(2*np.pi*FREQ1*t) + np.sin(2*np.pi*FREQ2*t)

T = -10 # desfase en muestras (debe ser entero, sino no funciona)

signal_fft = np.fft.fft(signal)
fshift = np.exp([-1j*2*np.pi*T*f/len(signal_fft) for f in range(0, len(signal_fft))])

new_signal = np.real(np.fft.ifft(fshift*signal_fft))
print(len(signal), len(new_signal))
plt.plot(t, signal, t, new_signal)
plt.show()

plt.plot(np.angle(signal_fft))
plt.show()
plt.close()

plt.plot(np.angle(signal_fft*fshift))
plt.show()
plt.close()
