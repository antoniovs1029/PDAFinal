import numpy as np

def calcular_seno(distancia, angulo, vsound = 343):
    return [0, - np.sin(np.radians(angulo))*distancia/vsound, np.sin(np.radians(angulo - 60))*distancia/vsound]

def calcular_perpendicular(distancia, angulo, vsound = 343):
    radio = (distancia*np.sqrt(3))/3
    cateto_y = np.sqrt(  radio**2 - (distancia/2)**2 )

    x1, y1 = distancia/2, cateto_y
    x2, y2 = -distancia/2, cateto_y
    x3, y3 = 0, -radio

    pendiente = np.tan(np.radians(-angulo + 90)) # Se debe poner asi pues las formulas son usando el angulo del eje x en sentido antihorario, pero el angulo de aire es desde el Y en sentido horario.

    def distancia_a_recta(x, y, pendiente):
        num = -pendiente*x + y
        denom = np.sqrt(pendiente**2 + 1)
        return num/denom

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

def calcular_exactas(dist_micros, angulo, vsound = 343, rad_fuente = 1):
    radio_micros = (dist_micros*np.sqrt(3))/3
    cateto_y = np.sqrt(  radio_micros**2 - (dist_micros/2)**2 )

    x1, y1 = dist_micros/2, cateto_y
    x2, y2 = -dist_micros/2, cateto_y
    x3, y3 = 0, -radio_micros

    angulo = np.radians(-angulo + 90) # Se debe poner asi pues las formulas son usando el angulo del eje x en sentido antihorario, pero el angulo de aire es desde el Y en sentido horario.

    fuente_x = rad_fuente*np.cos(angulo)
    fuente_y = rad_fuente*np.sin(angulo)

    def dist_puntos(x1, y1, x2, y2):
        return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )

    d1 = dist_puntos(x1, y1, fuente_x, fuente_y)/vsound
    d2 = dist_puntos(x2, y2, fuente_x, fuente_y)/vsound
    d3 = dist_puntos(x3, y3, fuente_x, fuente_y)/vsound

    return [d1 - d1, d2 - d1, d3 - d1]

def calcular_desfases(dist_micros, angulo, tipo, vsound = 343, rad_fuente = 1):
    if tipo == 'seno':
        return calcular_seno(dist_micros, angulo, vsound)
    elif tipo == 'perpendicular':
        return calcular_perpendicular(dist_micros, angulo, vsound)
    else:
        return calcular_exactas(dist_micros, angulo, vsound, rad_fuente)

if __name__ = "__main__":
    for angulo in [0, 90, 180, -90]:
        print("{0}°".format(angulo))
        print("\tSeno:", calcular_seno(.18, angulo))
        print("\tPerpendicular:", calcular_perpendicular(.18, angulo))
        print("\tExacta:", calcular_exactas(.18, angulo))
