# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Jesús Medina Taboada
"""

import numpy as np
import matplotlib.pyplot as plt
import math 

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

#--------- E J E R C I C I O    1.1  -------


#Funcion a estudiar
def E(u,v):
    return ((u**2*np.exp(v)-2*v**2*np.exp(-u))**2)   

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return (4*np.exp(-2*u)*(u**2*np.exp(u+v)-2*v**2)*(u*np.exp(u+v)+v**2))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return ((2*np.exp(-2*u))*(u**2*np.exp(u+v)-4*v)*(u**2*np.exp(u+v)-(2*v**2)))

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


	# Criterio de parada: numero de iteraciones y 
	# valor de f inferior a epsilon

def gradient_descent(w, eta, grad_fun, fun, error2get, maxIter):
    iterations = 0
    
    while iterations <= maxIter and fun(w[0],w[1]) >= error2get:
        w = w-eta*grad_fun(w[0],w[1])
        iterations += 1
    return w, iterations
     


eta = 0.01 
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])

w, it = gradient_descent(initial_point, eta, gradE, E, error2get, maxIter)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
input("\n--- Pulsar tecla para continuar ---\n")


#Segunda función a estudiar
def f(u,v):
    
	return u**2+2*v**2+2*math.sin(2*math.pi*u)*math.sin(2*math.pi*v)
	
# Derivada parcial de f respecto de u
def fu(u,v):

	return 2*u+4*math.pi*math.cos(2*math.pi*u)*math.sin(2*math.pi*v)

# Derivada parcial de f respecto de v
def fv(u,v):

	return 4*v+4*math.pi*math.sin(2*math.pi*u)*math.cos(2*math.pi*v)
	
# Gradiente de f
def gradf(u,v):
	return np.array([fu(u,v), fv(u,v)])



##############################################################################
##############################################################################

#--------- E J E R C I C I O    1.2  -------

# -----APARTADO A-----
# Usamos gradiente descendente con la función f para minimizarla.
# Los valores iniciales seran:
#           -Punto partida [1,1]
#           -Tasa de aprendizaje 0.01 y luego 0.1
#           -Máximo iteraciones 

def gra_desc_fun(w, eta, grad_fun, fun, maxIter=50):
	graf = [fun(w[0],w[1])]
	for k in range(1,maxIter):
		w = w-eta*grad_fun(w[0],w[1])
		graf.insert(len(graf),fun(w[0],w[1]))
				
	plt.plot(range(0,maxIter), graf, 'ro')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.show()	

print ('Resultados ejercicio 2\n')
print ('\nGrafica con tasa de aprendizaje igual a 0.01')
gra_desc_fun(np.array([0.1,0.1]) , 0.01, gradf, f)
print ('\nGrafica con tasa de aprendizaje igual a 0.1')
gra_desc_fun(np.array([0.1,0.1]) , 0.1, gradf, f)
input("\n--- Pulsar tecla para continuar ---\n")


# -----APARTADO B-----
# Obtener el valor minimo y los valores de las variables (x,y) se alcanzan
# cuando el punto de inicio se fija:
# (0,1, 0,1)
# (1, 1)
# (−0,5, −0,5)
# (−1, −1)

def gd(w, eta, grad_fun, fun, maxIter = 50):
	for i in range(0,maxIter):
		w = w-eta*grad_fun(w[0],w[1])
		
	return w

#Primer punto (0.1,0.1)
 
w = gd(np.array([0.1, 0.1]) , 0.01, gradf, f)
print ('Punto de inicio: (0.1, 0.1)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Minimo: ',f(w[0],w[1]))

input("\n--- Pulsar tecla para continuar ---\n")


#Segundo punto (1,1)

w = gd(np.array([1,1]) , 0.01, gradf, f)
print ('Punto de inicio: (1,1)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Minimo: ',f(w[0],w[1]))

input("\n--- Pulsar tecla para continuar ---\n")


#Tercer punto (-0.5,-0.5)

w = gd(np.array([-0.5,-0.5]) , 0.01, gradf, f)
print ('Punto de inicio: (-0.5,-0.5)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Minimo: ',f(w[0],w[1]))

input("\n--- Pulsar tecla para continuar ---\n")


#Cuarto punto (-1.0, -1.0)

w = gd(np.array([-1.0, -1.0]) , 0.01, gradf, f)
print ('Punto de inicio: (-1.0, -1.0)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Mínimo: ',f(w[0],w[1]))

input("\n--- Pulsar tecla para continuar ---\n")





###############################################################################
###############################################################################
###############################################################################
###############################################################################

#--------- E J E R C I C I O    2.1  -------

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Error(x,y,w):
	return (1/y.size)*np.linalg.norm(x.dot(w)-y)**2 

# Funcion del Gradiente Descendente Estocastico
def gde(x, y, eta, maxIter, minibatch):	
	w = np.zeros(len(x[0]), np.float64)
	posicion = np.array(range(0,x.shape[0]))
	
	for k in range(0,maxIter):
		w_2 = w
		for j in range(0,w.size):
			suma = 0			
			# Desordenamos las posiciones para el minibatch
			np.random.shuffle(posicion)
			tam_minibatch = minibatch
			# Hacemos sumatoria de la función
			suma = (np.sum(x[posicion[0:tam_minibatch:1],j]*
                  (x[posicion[0:tam_minibatch:1]].dot(w_2) -
                   y[posicion[0:tam_minibatch:1]])))
			
			w[j] -= eta*(2.0/tam_minibatch)*suma
		
	return w

# Funcion de Pseudoinversa	
def pseudoinversa(x, y):
    
	u,d,vt = np.linalg.svd(x)
	d_inversa = np.linalg.inv(np.diag(d))
	v = vt.transpose()
    
	# Calculo de la pseudoinversa de x
	x_inv = v.dot(d_inversa).dot(d_inversa).dot(v.transpose()).dot(x.transpose())
	w = x_inv.dot(y)
	return w


# Leemos datos de para el test y lo sde entrenamiento
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')


#Llamamos a la funcion del gradiente descendente estocastico
w = gde(x, y, 0.01, 500, 64)

print ('La bondad del resultado obtenido para el la funcion gradiente\
       descendiente estocastico a través de Ein y Eout es:\n')
print ("Ein:", Error(x,y,w))
print ("Eout:", Error(x_test, y_test, w))



# Hacemos el grafico teniendo en cuenta las columnas 1 y 2 de la variable x
# que son la intensidad y la simetria. Por otra parte separamos los valores de
# y por la clase a la que pertenezcan

plt.scatter(x[:,1],x[:,2], c=y)
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.title('Gradiente Descendente Estocastico')
plt.ylabel('Simetria')
plt.xlabel('Intensidad')
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")


#Llamamos a la funcion de la pseudoinversa
w = pseudoinversa(x, y)

print ('La bondad del resultado obtenido para el la funcion pseudoinversa\
       a través de Ein y Eout es:\n')
print ("Ein: ", Error(x,y,w))
print ("Eout: ", Error(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

plt.scatter(x[:,1],x[:,2], c=y)
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.title('Pseudoinversa')
plt.ylabel('Simetria')
plt.xlabel('Intensidad')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

##############################################################################
##############################################################################

#--------- E J E R C I C I O    2.2  -------

print('Ejercicio 2\n')

# Funcion simula_unif(N, 2, size) que nos devuelve N coordenadas 2D de 
#puntos uniformemente muestreados dentro del cuadrado definido por
# [−size, size] × [−size, size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))
	
# EXPERIMENTO	
    
# -----APARTADO A-----
# Generamos una muestra de entrenamiento con N = 1000 puntos en el cuadrado
# X = [−1, 1] × [−1, 1] y lo pintamos en un mapa de puntos 2D


plt.title('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')
x = simula_unif(1000, 2, 1)
plt.scatter(x[:,0],x[:,1],c='red')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# -----APARTADO B-----
# Creamos una funcion sing para devolver 1 o -1 segun el valor que se recibe
# como argumento.

# Funcion signo
def sign(x):
	if x >= 0:
		return 1
	return -1
# Creamos f2 que se encarga de asignar etiquetas a la muestra x
def f2(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

# Introducimos 10% de ruido a traves de un array con 10% de aletoriedad
# que simula la introducción de ruido
p = np.random.permutation(range(0,x.shape[0]))[0:x.shape[0]//10]

# Ordenamos el array obtenido
p.sort()
j = 0
y = []

for i in range(0,x.shape[0]):
	# Si i está en p cambiamos el signo si no lo mantenemos
	if i == p[j]:
		j = (j+1)%(x.shape[0]//10)
		y.append(-f2(x[i][0], x[i][1]))
	else:
		y.append(f2(x[i][0], x[i][1]))

x = np.array(x, np.float64)
y = np.array(y, np.float64)


plt.title('Muestra N = 1000, cuadrado [-1,1]x[-1,1] con muestras etiquetas\
 y 10% de ruido en ellas')
plt.scatter(x[:,0],x[:,1], c=y)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


# -----APARTADO C-----
# Usando como vector de características (1, x1, x2) ajustar un modelo de regresion
#lineal al conjunto de datos generado y estimar los pesos w. Estimar el error de
#ajuste Ein usando Gradiente Descendente Estocástico (SGD)

# Creamos array de unos para la regresion lineal (1, x0, x1)
q = np.array([np.ones(x.shape[0], np.float64)])
x = np.concatenate((q.T, x), axis = 1)

w = gde(x, y, 0.01, 1000, 64)

print ('Error de ajuste Ein usando Gradiente Descediente estocastico')
print ('con 1000 iteraciones')
print ("\n\nEin: ", Error(x,y,w))
plt.scatter(x[:,1],x[:,2], c=y)
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.axis([-1,1,-1,1])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")



# -----APARTADO D-----
# Ejecutar todo el experimento definido por (a)-(c) 1000 veces (generamos 1000
# muestras diferentes) y:
#    • Calculamos el valor medio de los errores Ein de las 1000 muestras.
#    • Generamos 1000 puntos nuevos por cada iteración y calcular con ellos el
#      valor de Eout en dicha iteración. Calcular el valor medio de Eout en
#      todas las iteraciones.

#Declaramos las variables de la media de Ein y Eout
Ein = 0
Eout = 0 
    
for k in range(0,1000):
	x = simula_unif(1000, 2, 1)
    
	# Array de unos para la regresion lineal (1, x0, x1)
	q = np.array([np.ones(x.shape[0], np.float64)])
	x = np.concatenate((q.T, x), axis = 1)
	y = []
	
	# Array con 10% de indices aleatorios para introducir ruido
	b = np.random.permutation(range(0,x.shape[0]))[0:x.shape[0]//10]
	b.sort()
	j = 0
	
	for i in range(0,x.shape[0]):
		if i == b[j]:
			j = (j+1)%(x.shape[0]//10)
			y.append(-f2(x[i][1], x[i][2]))
		else:
			y.append(f2(x[i][1], x[i][2]))
	
	y = np.array(y, np.float64)	
	
	# Solo 10 iteraciones y minibatch de 32 para que la ejecucion
	# no tarde demasiado tiempo
	w = gde(x, y, 0.01, 10, 32)
	
	# Simulamos los datos del test para despues calcular el Eout
	x_test = simula_unif(1000, 2, 1)	
	x_test = np.concatenate((q.T, x_test), axis = 1)
	

    
	Ein += Error(x,y,w)
	Eout += Error(x_test,y,w)

Ein /= 1000
Eout /= 1000

print ('Valores medios de Ein y Eout medios  tras repetir el experimento 1000 veces:\n')
print ("Ein media: ", Ein)
print ("Eout media: ", Eout)




