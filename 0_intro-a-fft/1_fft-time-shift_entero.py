import numpy as np
import matplotlib.pyplot as plt

SAMPLING_RATE = 128.0 # 128 puntos (samples) por segundo. Ojo: a veces se usa el término frame en lugar de sample, pero es un desmadre.
t = np.arange(0, 1, 1/SAMPLING_RATE) # eje del tiempo, durará 1 segundo

FREQ1 = 5
FREQ2 = 2.5

signal = np.sin(2*np.pi*FREQ1*t) + np.sin(2*np.pi*FREQ2*t)

plt.plot(signal)
plt.title("Señal a transformar")
plt.xlabel("Punto #")
plt.show()
plt.close()

plt.plot(t, signal)
plt.title("Señal a transformar (con eje en el tiempo)")
plt.xlabel("Tiempo (segundos)")
plt.show()
plt.close()

signal_fft = np.fft.fft(signal)

print("Tamaño de la señal original: {0}\nTamaño de la señal transformada: {1}".format(len(signal), len(signal_fft))) # Son del mismo tamaño, solo que signal_fft es un arreglo de números complejos

# Como no se puede plotear el arreglo de complejos en el plano, se suele plotear el absoluto de cada elemento.
# En algunas otras ocasiones interesa más plotear la fase de cada elemento, pero es menos común.

plt.plot(np.abs(signal_fft))
plt.title("Absoluto de la transformada")
plt.xlabel("Punto #")
plt.show()
plt.close()

plt.plot(np.angle(signal_fft))
plt.title("Fase de la transformada")
plt.xlabel("Punto #")
plt.ylabel("Angulo (radianes)")
plt.show()
plt.close()

# La transformada está en el dominio de la frecuencia.
# Sin embargo la FFT realmente no funciona pensando en tiempo y frecuencia, sino simplemente tomando arreglos de cierto tamaño
# y devolviendo arreglos del mismo tamaño.
# Para expresar el eje de la frecuencia se puede usar la función fftreq
# Se nota que la FFT devuelve, como es correcto, un arreglo "reflejado" donde primero se ponen frecuencias positivas
# y después frecuencias negativas.

frecs = np.fft.fftfreq(len(signal_fft), 1/SAMPLING_RATE) # regresa los verdaderos indices (las frecuencias)
plt.plot(frecs, np.abs(signal_fft)) # al plotearlo asi, se ordena a la señal como debe de ser
plt.title("Absoluto de la transformada")
plt.xlabel("Frecuencia")
plt.show()
plt.close()

# Se nota que como la señal tenía un SAMPLING RATE de 128, el arreglo tiene 63 frecuencias positivas, 64 negativas y el cero.
# Y que entonces la máxima frecuencia positiva es 63.

# En señales de audio, típicamente son de 48000Hz, luego la máxima frecuencia es de ~ 24000.

# Finalmente calculo la inversa y veo que da lo mismo que la original:
new_signal = np.real(np.fft.ifft(signal_fft))
plt.plot(t, signal, t, new_signal) # se sobrelapan
plt.title("Señal tras la Inversa")
plt.show()
plt.close()
