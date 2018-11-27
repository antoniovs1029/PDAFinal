"""
Calcular desfases!
"""

import numpy as np
import matplotlib.pyplot as plt

def distancia_a_recta(x, y, pendiente):
    num = -pendiente*x + y
    print(num)
    denom = np.sqrt(pendiente**2 + 1)
    return num/denom

def plotear(distancia, angulo):
    radio = (distancia*np.sqrt(3))/3
    cateto_y = np.sqrt(  radio**2 - (distancia/2)**2 )
    print("Radio: {0}, Cateto_Y: {1}".format(radio, cateto_y))

    x1, y1 = distancia/2, cateto_y
    x2, y2 = -distancia/2, cateto_y
    x3, y3 = 0, -radio

    print("X{0}: {1}, Y{0}: {2}".format(1, x1, y1))
    print("X{0}: {1}, Y{0}: {2}".format(2, x2, y2))
    print("X{0}: {1}, Y{0}: {2}".format(3, x3, y3))

    circ_x = [radio*np.cos(t) for t in np.arange(0,2*np.pi,.1)]
    circ_y = [radio*np.sin(t) for t in np.arange(0,2*np.pi, .1)]

    pendiente = np.tan(np.radians(angulo))
    pendiente_inversa = None

    if pendiente == 0:
        d1 = x1
        d2 = x2
        d3 = x3
    else:
        pendiente_inversa = -1 / pendiente
        d1 = distancia_a_recta(x1, y1, pendiente_inversa)
        d2 = distancia_a_recta(x2, y2, pendiente_inversa)
        d3 = distancia_a_recta(x3, y3, pendiente_inversa)

    print("Pendiente: {0}".format(pendiente))
    if pendiente_inversa: print("Pendiente Inversa:", pendiente_inversa)
    print("D1: {0}, D2: {1}, D3: {2}".format(d1,d2,d3))

    plt.scatter(circ_x, circ_y, c = 'blue')
    plt.scatter([x1, x2, x3], [y1, y2, y3], c = 'red')
    plt.plot([-distancia, distancia], [-pendiente*distancia, pendiente*distancia], c = 'green')
    if pendiente_inversa is not None:
        plt.plot([-distancia, distancia], [-pendiente_inversa*distancia, pendiente_inversa*distancia], c = 'brown')
    plt.scatter([0], [0], c= 'black')

    plt.show()

distancia = .18
angulo = 110
plotear(distancia, angulo)
