# Cuando el desfase no es un numero entero, uno debe calcular bien las frecuencias a multiplicar
# Basado en: https://www.mathworks.com/matlabcentral/answers/21883-fractional-delay-using-fft-ifft

import numpy as np
import matplotlib.pyplot as plt

SAMPLING_RATE = 128.0 # 128 puntos por segundo. En general *parece* necesario que la señal a shiftear tenga como tamaño a una potencia de 2
t = np.arange(0, 1, 1/SAMPLING_RATE) # eje del tiempo, durara 1 segundo

FREQ1 = 5
FREQ2 = 2.5

signal = np.sin(2*np.pi*FREQ1*t) + np.sin(2*np.pi*FREQ2*t)

T = -.5 # desfase en frames

signal_fft = np.fft.fft(signal)
frecs = list(range(0, len(signal_fft)//2 + 1)) + list(range(-len(signal_fft)//2 + 1, 0)) # solo funciona si len(signal_fft) es par... si no, hay que añadir algo en medio...
fshift = np.exp([-1j*2*np.pi*T*f/len(signal_fft) for f in frecs])

new_signal = np.real(np.fft.ifft(fshift*signal_fft))
plt.plot(t, signal, t, new_signal)
plt.show()

plt.plot(np.angle(signal_fft))
plt.show()
plt.close()

plt.plot(np.angle(signal_fft*fshift))
plt.show()
plt.close()
