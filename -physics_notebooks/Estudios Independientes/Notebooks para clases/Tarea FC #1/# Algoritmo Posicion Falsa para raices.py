# Algoritmo Posicion Falsa para raices
# busca en intervalo [a,b]
import numpy as np

# INGRESO
fx = lambda x: (1/9)+(225**2)*((6/10)*(10**(-6))*x - ((1/2)*x)**(-1))**2 - 1

a = 1
b = 1000
tolera = 0.001

# PROCEDIMIENTO
tramo = abs(b-a)
while not(tramo<=tolera):
    fa = fx(a)
    fb = fx(b)
    c = b - fb*(a-b)/(fa-fb)
    fc = fx(c)
    cambia = np.sign(fa)*np.sign(fc)
    if (cambia > 0):
        tramo = abs(c-a)
        a = c
    else:
        tramo = abs(b-c)
        b = c
raiz = c

# SALIDA
print(raiz)