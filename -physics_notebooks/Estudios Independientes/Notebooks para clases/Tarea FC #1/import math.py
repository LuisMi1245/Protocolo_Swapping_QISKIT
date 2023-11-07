import math
#Rutina Newton Raphson.
def Newton_Raphson(f, Derivatef, raiz_aprox, tolerancia, max_iteraciones):
    xn = x_aprox
    for n in range(1 , max_iteraciones+1):
        f_xn = f(xn)
        if abs(f_xn) < tolerancia:
            print('Solución para',n,'iteraciones')
            return xn
        Df_xn = Derivatef(xn)
        if Df_xn == 0:
            print('Solución no encontrada.')
            return None
        xn = xn - f_xn/Df_xn
    print('Máximo de iteraciones superadas')
    return None

#Función utilizada para solucionar el problema.
f = lambda h: (h**2)*((3*(3)-h) - (150/math.pi)

#La respectiva función derivada.
Df = lambda h: -3*h**2 + 18*h

#Se ejecuta la función con una raiz aproximada de 2.77, 3 iteraciones y épsilon de 1e-07
newton(f, Df, 2.77, 1e-6,3)