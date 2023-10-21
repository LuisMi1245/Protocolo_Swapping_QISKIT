import math as mt
import numpy as np

def newInterpolation(x, y, val_x, epsilon):
    
    #Comprobamos que la cantidad de datos del vector X, es igual al vector Y:
    if len(x)==len(y):
        
        #Establecemos los n grados de interpolación. La función len(x) 
        #nos arroja la longitud del arreglo.
        
        n = len(x) - 1
        N = n + 1
        
        #Imprimimos los vectores X e Y en formato de tabla:
        print("--------------------------------------------")
        print("Número de puntos:", N, "\n")
        print('{:^12}{:^12}{:^12}'.format("i","Xi", "Yi"))
        print("--------------------------------------------")
        for i in range(0,N):
            print('{:^12}{:^12}{:^12}'.format(i, x[i], y[i]))
        print("--------------------------------------------\n")
        
        
        #Establecemos la forma de las matrices de las diferencias divididas "fdd",
        # errores absolutos "ea", y los valores de la función para 
        # cada grado de interpolación "yint".
        
        print("--------------------------------------------")
        print("Interpolación para x =", val_x, "\n")
        fdd = np.zeros((N,N))
        ea = np.zeros(N)
        yint = np.zeros(N)
        yint2 = 0.0
        xterm = 1.0

        #Asignamos al grado 0 de la matriz de las diferencias divididas,
        # los valores de Yi proporcionados por el usuario.
        for i in range(0,N):
            fdd[i,0] = y[i]
        
        #Calculamos las diferencias divididas para los grados [1,n], y por tanto los
        # coeficientes del polinomio de interpolación de Newton, teniendo en cuenta 
        # el funcionamiento de los ciclos for de Python.
        for j in range(1,N):
            for i in range(0,N-j):
                fdd[i,j] = (fdd[i+1,j-1]-fdd[i,j-1])/(x[i+j] - x[i])
        
        #Asignamos el primer valor de la función para interpolación de grado 0:"
        yint[0] = fdd[0,0]

        #Calculamos los términos del polinomio para grados mayores, utilizando
        # el valor de evaluación val_x proporcionado por el usuario.
        for orden in range(1,N):
            xterm = xterm*(val_x - x[orden-1])
            yint2 = yint[orden-1] + fdd[0,orden]*xterm
            ea[orden-1] = abs(yint2 - yint[orden-1])
            
            #Establecemos un criterio de paro para cierta tolerancia:
            if(ea[orden-1] <= epsilon): 
                break
            yint[orden] = yint2
        
        #Imprimimos los valores de la función para cada grado de interpolación
        # con su error respectivo:
        print('{:^12}{:^12}{:^12}'.format("GRADO","F(x)", "ERROR"))
        print("--------------------------------------------")
        for i in range(0,N):
            print('{:^12}{:^12}{:^12}'.format(i, yint[i], ea[i]))
        print("--------------------------------------------\n")
        
        #El valor de la función evaluada en X, obtenido por interpolación es:
        print("F"+"("+str(val_x)+")"+" =", yint2)
    

newInterpolation((2,4,6,8,10), (4,16,36,64,100), 5, 0e-5)