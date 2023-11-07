#########################################################################################
#         CALCULO DE LA RADIACIÓN ESPECTRAL DE CUERPO NEGRO                             #
#########################################################################################

# Se cargan bibliotecas habituales:
import numpy as np  # Herramienta para cálculo numérico
import pandas as pd  # Herramienta para manipulación y visualización de tablas
from tabulate import tabulate

def main():
    # Se definen las constantes físicas.
    h = 6.626176e-34  # Constante de Planck en J*s
    c = 2.997923e+8  # Velocidad de la luz en m/s
    k = 1.380662e-23  # Constante de Boltzmann en J/K

    # Función que representa la densidad de energía de la radiación de cuerpo negro:
    def densidad_energia(lam, T):
        return (2 * h * np.pi * (c**2)) / ((lam**5) * (np.exp((h * c) / (k * lam * T)) - 1))

    # Temperatura de entrada en grados Kelvin:
    temps = float(input("Ingrese Temperatura en Kelvin: "))

    # Se crea arreglo de valores de longitudes de onda en µm:
    rango = [float(input("Ingrese rango inicial en µm")),\
             float(input("Ingrese rango final en µm")),\
             float(input("Ingrese tamaño de paso en µm"))]

    lam = np.arange(rango[0], rango[1]+rango[2], rango[2]) * 1e-6

    # Se calcula densidad de energía en función de la temperatura
    # y el rango de longitudes de onda de entrada:
    dns = densidad_energia(lam, temps)

    # Elaboramos una tabla de valores de: longitud de onda vs densidad espectral:
    tabla = {"Longitud de Onda (µm)": lam*1e+6, "Densidad Espectral (W/m² µm)": dns/1e+6}
    print(tabulate(tabla, headers="keys", tablefmt="psql"))

    return [temps, rango, lam, dns]

# Se grafica la curva de densidad de energía:
import matplotlib.pyplot as plt  # Herramienta para realizar gráficas
fig, ax = plt.subplots()
#plt.xlim([0, 100])
plt.ylim([0, 80])
plt.xlabel(r'Longitud de onda ($\mu m$)')
plt.ylabel(r'Densidad espectral ($W/m^{2}\mu m$)')
plt.title(r'Densidad Espectral de los 3 Cuerpos Celestes')
plt.grid()


from scipy import integrate
temps_legend = []
for i in range(3):
    temps, rango, lam, densidades = main()
    temps_legend.append("T = "+str(temps)+"K")
    ax.plot(lam*1e+6, densidades/1e+6, "o")

    # Se calcula área bajo la curva de densidad de energía
    area = integrate.simpson(densidades, lam)
    print("Energía Irradiada por unidad de área y tiempo: " + str(area) + " W/m^2")
    # Se halla la longitud de onda para el máximo de densidad de energía:
    a = 2.8996e-3  # las unidades son: m*K
    lammax = (a / temps) * (1e+6)  # Realizamos la conversión a µm.
    print("Longitud de onda para máximo de densidad de energía: " + str(lammax) + "µm")

plt.legend(temps_legend)
plt.show()