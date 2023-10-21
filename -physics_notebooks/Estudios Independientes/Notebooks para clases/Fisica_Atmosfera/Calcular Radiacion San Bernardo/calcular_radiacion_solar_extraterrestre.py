#########################################################################################
#                        Nombre del programa:                                           #
#               Cálculo de la radiación solar extraterrestre                            #
#               Luis Miguel Patiño, Jose Sebastian Molina                               #
#########################################################################################
from numpy import arcsin, sin, cos, pi
import numpy as np

def angulo_elevacion_solar(latitud_geog, angulo_declinacion,
                           angulo_horario):  # Algoritmo para hallar la elevación solar en radianes
    return arcsin(
        sin(latitud_geog) * sin(angulo_declinacion) + cos(latitud_geog) * cos(angulo_declinacion) * cos(angulo_horario))

def angulo_declinacion_solar(dia_juliano):  # Ecuación de declinación solar en unidades de grados
    return 23.45 * sin((2 * pi * (284 + dia_juliano)) / 365)


def angulo_horario(t_solar):  # Ecuación del ángulo horario en unidades de radianes
    return (pi * (t_solar - 12)) / 12

def tiempo_solar(t_local, Eqt, longMS, longLOCAL):  # Ecuación del tiempo solar en unidades de horas
    return t_local + (Eqt / 60) + ((longMS - longLOCAL) / 15)

def Eqt(dia_juliano):  # Algoritmo para hallar ecuación del tiempo en unidades de minutos
    if (1 <= dia_juliano <= 106):
        return -14.2 * sin((pi * (dia_juliano + 7)) / 111)
    elif (247 <= dia_juliano <= 365):
        return 16.4 * sin((pi * (dia_juliano - 247)) / 113)

def radiacion_solar(dia_juliano, gamma):  # Algoritmo para hallar la radiación solar en W/m^2
    s0 = 1367  # Constante solar unidades de W/m²
    B = lambda d: (2 * pi * d) / 365
    f = 1.00011 + 0.034221 * cos(B(dia_juliano)) + 0.00128 * sin(B(dia_juliano)) + 0.000719 * cos(
        2 * B(dia_juliano)) + 0.000077 * sin(2 * B(dia_juliano))
    return s0 * f * sin(gamma)


def main():
    a = pi / 180  # Factor de Conversión a radianes
    b = 1 / a  # Factor de Conversión a grados
    print("***************************************************")
    print("* Calcular la radiación solar extraterrestre *")
    print("***************************************************")

    pais = input("Nombre del país: ")
    longMS = float(input(f"Introduzca la longitud decimal del meridiano estándar de {pais} (°): "))

    municipio = input("Nombre del municipio: ")
    longLOCAL = float(input(f"Introduzca la longitud decimal local de {municipio} (°): "))
    latitud_geografica = float(input(f"Introduzca la latitud geográfica decimal de {municipio} (°): "))
    dias = [int(input("Introduzca el día juliano del año #1: ")), int(input("Introduzca el día juliano del año #2: "))]
    tiempo_local = int(input("Introduzca la hora (horas): "))

    print("")
    print(f"*** RESULTADOS PARA {municipio}, {pais} ***")

    for dia in dias:
        delta = angulo_declinacion_solar(dia)
        eqt = Eqt(dia)
        t_solar = tiempo_solar(tiempo_local, eqt, longMS, longLOCAL)
        omega = b * angulo_horario(t_solar)
        gamma = b * angulo_elevacion_solar(a * latitud_geografica, a * delta, a * omega)
        radiacion = radiacion_solar(dia, a * gamma)

        print(f"Día {dia}, {tiempo_local} horas")
        print("------------------------------------------")
        print(f"1). Ángulo de declinación solar: {delta:.3f}°")
        print(f"2). Ángulo Horario: {omega:.3f}°")
        print(f"3). Ángulo de Elevación Solar: {gamma:.3f}°")
        print(f"4). Radiación Solar Extraterrestre: {radiacion:.3f} W/m²")
        print("------------------------------------------")
        print("")


main()