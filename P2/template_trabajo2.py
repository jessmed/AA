# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Jesus Medina Taboada
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

print ('\nEJERCICIO 1\n')
print ('Simula_unif')
x = simula_unif(50, 2, [-50,50])
plt.scatter(x[:, 0], x[:, 1])
plt.show()

print ('Simula_gaus')
x = simula_gaus(50, 2, np.array([5,7]))
plt.scatter(x[:, 0], x[:, 1])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

	
	
# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def f_(X,a,b):
	return X[:, 1]-a*X[:, 0]-b

# simulamos una muestra de puntos 2D 
x = simula_unif(50,2, [-50, 50])

# simulamos una recta
intervalo=[-50,50]
[a,b] = simula_recta(intervalo)

# agregamos las etiquetas usando el signo de la funcion f
y = []

for i in range(0,x.shape[0]):
    y.append(f(x[i][0], x[i][1], a, b))
    

# Funcion que dibuja una gráfica de los datos etiquetados y la recta

def plot_datos_recta(x, y, a, b, title='Point clod plot', xaxis='x axis', yaxis='y axis'):
    # Primero transformamos lo parámetros de la recta en coeficientes de w
    # siendo a la pendiente de la recta y b el término independeinte   
    w = np.zeros(3, np.float64)
    w[0] = -a
    w[1] = 1.0
    w[2] = -b
    
    # Preparar datos 
    min_xy = x.min(axis=0)
    max_xy = x.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    # Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    # Generamos puntos(plot)
    f, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)

    ax.scatter(x[:, 0], x[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0)
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1])
    )
    plt.show()


plot_datos_recta(x, y, a, b)


input("\n--- Pulsar tecla para continuar ---\n")


# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

# Introducimos ruido en las etiquetas

# Vector de etiquetas 1 y -1
i_pos = []   #1
i_neg = []   #-1

# Vector de etiquetas que vamos a modificar
i_aux = np.array(y)

for i in range(0,len(y)):
	if y[i] == 1:
		i_pos.append(i)
	else:
		i_neg.append(i)

i_pos = np.array(i_pos)
i_neg = np.array(i_neg)
	
# Desordenamos las posiciones de los vectores
np.random.shuffle(i_pos)
np.random.shuffle(i_neg)

# Modificamos 10% de las etiquetas positivas
for i in range(0,i_pos.size//10):
	i_aux[i_pos[i]] = -1

# Modificamos 10% de las etiquetas negativas
for i in range(0,i_neg.size//10):
	i_aux[i_neg[i]] = 1	

# Dibujamos la grafica
plot_datos_recta(x, i_aux, a, b)

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la 
# frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
    
def f1(X):
	return (X[:,0]-10)**2+(X[:,1]-20)**2-400

def f2(X):
	return 0.5*(X[:,0]+10)**2+(X[:,1]-20)**2-400
	
def f3(X):
	return 0.5*(X[:,0]-10)**2-(X[:,1]+20)**2-400

def f4(X):
	return X[:,1]-20*X[:,0]**2-5*X[:,0]+3

print('\nf(x,y) = (x-10)^2+(y-20)^2-400\n')
plot_datos_cuad(x, i_aux, f1)

print('\nf(x,y) = 0.5*(x+10)^2+(y-20)^2-400\n')
plot_datos_cuad(x, i_aux, f2)

print('\nf(x,y) = 0.5*(x-10)^2-(y+20)^2-400\n')
plot_datos_cuad(x, i_aux, f3)

print('\nf(x,y) = y-20x^2-5x+3\n')
plot_datos_cuad(x, i_aux, f4)

input("\n--- Pulsar enter para continuar ---\n")

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
	w = vini
	x = np.concatenate((datos, np.ones((datos.shape[0], 1), np.float64)), axis=1)
	iters = 0
	while iters < max_iter:			
		stop = True
		iters+=1
		for i in range(0,len(label)):
			if np.sign(w.dot(x[i])) != label[i]:
				w = w + label[i]*x[i]
				stop = False
		if stop:
			break				
	return w, iters	

#Apartado a.1) con el vector de ceros
vini = np.zeros(x.shape[1]+1, np.float64)
w, iters = ajusta_PLA(x, y, 500, vini)
print ('Media de iteracione para converger con vector cero:', iters)



#Apartado a.2) con vectores de números aleatorios en [0, 1] (10 veces)
suma = 0   
# Random initializations
for i in range(0,10):
	vini = simula_unif(1, 3,[0,1])	
	w, iters = ajusta_PLA(x, y, 500, vini)
	suma = suma + iters     
    
    
print('Valor medio de iteraciones necesario para converger: ', suma/10)

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

vini = np.zeros(x.shape[1]+1, np.float64)
w, iters = ajusta_PLA(x, i_aux, 500, vini)
print (' Iteraciones con vector cero:', iters)

suma = 0

for i in range(0,10):
	vini = np.random.uniform(0, 1, x.shape[1]+1)	
	w, iters = ajusta_PLA(x, i_aux, 500, vini)
	suma = suma + iters 
	
print (' Iteraciones vector con aleatorios: ', suma/10)



input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

print('\nEJERCICIO 2\n')

def sigmoide(x):
	return 1/(np.exp(-x)+1)

# Regresion logistica Gradiente Descendente Estocastico
def rl_sgd(X, y, max_iters, tam_minibatch, lr = 0.01, epsilon = 0.01):		
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	w = np.zeros(len(x[0]), np.float64)	
	index = np.array(range(0,x.shape[0]))
	tam = tam_minibatch
	
	for k in range(0,max_iters):
		w_old = np.array(w)

		np.random.shuffle(index)		
		for j in range(0,w.size):			
			
			suma = np.sum(-y[index[0:tam:1]]*x[index[0:tam:1],j]*
			(sigmoide(-y[index[0:tam:1]]*(x[index[0:tam:1]].dot(w_old)))))
			
			w[j] -= lr*suma
			
		if np.linalg.norm(w-w_old) < epsilon:
			break
		
	return w

a, b = simula_recta(intervalo=(0,2))	
X = simula_unif(100, 2, (0,2))
y = np.sign(f_(X,a,b))


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).  
def Err(X,y,w):	
	tam = X.shape[0]	
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	return (1.0/tam)*np.sum(np.log(1+np.exp(-y[0:tam:1]*(x[0:tam:1].dot(w)))))
	
def reetiquetar(y):
	y_ = np.array(y)

	for i in range(0, y_.size):
		if y_[i] == -1:
			y_[i] = 0
			
	return y_
	
def error_acierto(X,y,w):		
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	tam = y.size
	suma = 0
	
	for i in range(0,tam):
		if np.abs(sigmoide(x[i].dot(w))-y[i]) > 0.5:
			suma += 1
			
	return suma/tam	
	
w = rl_sgd(X ,y, 1000, 64)

y_ = reetiquetar(y)

print ('\nEin:',error_acierto(X,y_,w))	
	
X_test = simula_unif(2000, 2, (0,2))
y_test = np.sign(f_(X_test,a,b))	

y_test_ = reetiquetar(y_test)

print ('\nEout:',error_acierto(X_test,y_test_,w))


###############################################################################
###############################################################################
