"""
Filtro pasoaltas, que no usa scipy para leer y escribir los wavs,
por lo que complica varias cosas

Basado en http://pythonforengineers.com/audio-and-digital-signal-processingdsp-in-python/
"""

import wave
import struct
from math import sin, pi
import numpy as np
import matplotlib.pyplot as plt

num_samples = 48000
frame_rate = 48000.0
file_in = "wav_mic1.wav"

filter_freq = 1000

data_out = np.array([])
with  wave.open(file_in, 'r') as wav_file:
    print(wav_file.getparams())
    input()
    for _ in range(int(wav_file.getnframes() / num_samples)):
        data = wav_file.readframes(num_samples)
        data = struct.unpack('{n}h'.format(n = num_samples), data)
        data = np.array(data)
        data_fft = np.fft.fft(data)
    
        for i in range(len(data_fft)):
            if i < filter_freq or len(data_fft) - filter_freq < i:
                data_fft[i] = 0

        data = np.real(np.fft.ifft(data_fft))
        data_out = np.hstack([data_out, data])

plt.plot(data_out)
plt.show()

nframes = len(data_out)
comptype = "NONE"
compname = "not compressed"
nchannels = 1
sampwidth = 2
out_file = "test_filtrado.wav"
wav_file = wave.open(out_file, 'w')
wav_file.setparams((nchannels, sampwidth, int(frame_rate), nframes, comptype, compname))

for s in data_out:
    wav_file.writeframes(struct.pack('h', int(s))) #el pack 'h' lo convierte en hexadecimal, que es como debe escribirse en el wav

wav_file.close()
