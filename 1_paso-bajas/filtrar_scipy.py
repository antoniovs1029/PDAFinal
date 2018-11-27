"""
Filtro pasobajas, que deja pasar las frecuencias de la señal hasta una frequencia específicada

Toma como entrada un archivo wav de 1 canal (como los archivos de AIRA) y escribe como salida
otro archivo wav de 1 canal con el mismo sample rate que el primero.

Para "simular" que se hace como lo hacemos en C, toma el audio original y lo divide en ventanas,
luego un ciclo itera sobre cada ventana y la filtra. Las ventanas filtradas se van añadiendo al
arreglo "output_audio" que será el que se escribirá completo al archivo de salida.

No aplico hahn, pero se podría hacer tal y como lo hicimos en clase. De cualquier forma, suena
bien.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

FILTER_FREQ = 1000 # Maxima frecuencia que se dejara pasar
WINDOW_SIZE = 1024 # Tamaño de la ventana para simular "tiempo real", a mayor tamaño, mayor calidad del filtrado

file_in = "wav_mic1.wav"
file_out = "filtrado.wav"

sample_rate, input_audio = wavfile.read(file_in) # se lee el audio y su sample_rate (tipicamente 48000)
print("Forma del audio de entrada:", input_audio.shape, "Tipo de dato:", input_audio.dtype) # lo lee como con tipo de dato int16

input_windows = np.array_split(input_audio, len(input_audio) / WINDOW_SIZE) # se separa en ventanas
output_audio = np.array([], dtype = np.int16) # es necesario que la salida tenga tipo de dato int16, sino se escucha como con mucho ruido

for window in input_windows:
    window_fft = np.fft.fft(window)

#    print(len(window), np.mean(window), np.max(window), np.min(window))
#    plt.plot(window)
#    plt.show()
#    plt.close()

    for i in range(len(window_fft)):
        imin = FILTER_FREQ / (sample_rate / WINDOW_SIZE)
        if imin < i and i < WINDOW_SIZE - imin:
            window_fft[i] = 0

#    plt.plot(np.abs(window_fft))
#    plt.show()
#    plt.close()

    output_window = np.real(np.fft.ifft(window_fft))

    # Se añade la ventana de salida al audio que escribiremos, debemos asegurar que sea de tipo int16
    output_window = output_window.astype(np.int16)
    output_audio = np.concatenate((output_audio, output_window))
    
wavfile.write(file_out, sample_rate, output_audio) # Se escriben todas las ventanas de salida que guardamos
print("Forma del audio de salida:", output_audio.shape, "Tipo de dato:", output_audio.dtype) # tanto la forma como el tipo de dato debería coincidir con el de entrada
