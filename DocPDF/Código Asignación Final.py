import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Primero vamos a importar los datos:
df = pd.read_csv('masas.csv')
df

#PRIMER PUNTO 

#PUNTO A.1 MOMENTO DE INERCIA ORNDEN 0
# Los datos de las masas
segunda_columna = df.columns[1]

# Sumar los valores de la segunda columna
momentocero = df[segunda_columna].sum()

#PUNTO A.2 MOMENTO DE INERCIA ORDEN 1
# Extraer columnas
m = df['masas']
x = df['x']
y = df['y']

masa_total = m.sum()

# Calcular centro de masa
x_cm = (m * x).sum() / masa_total
y_cm = (m * y).sum() / masa_total

#PUNTO A.3 MOMENTO DE INERCIA ORDEN 2
# Calcular momentos de inercia

# Coordenadas relativas al centro de masa
x_rel = x - x_cm
y_rel = y - y_cm

# Componentes del tensor de inercia
I_xx = (m * y_rel**2).sum()
I_yy = (m * x_rel**2).sum()
I_xy = -(m * x_rel * y_rel).sum()

# Tensor de inercia 2x2
I_tensor = np.array([[I_xx, I_xy],
                     [I_xy, I_yy]])

#PROBLEMA DE AUTOVALORES Y AUTOVECTORES PARA NUESTRA MATRIZ 2X2 DE LOS MOMENTOS DE INERCIA

autovalores, autovectores = np.linalg.eigh(I_tensor)
#Autovectores desde la matriz (columnas)
Eje_principal_1 = autovectores[:, 0]  # Primer autovector
Eje_principal_2= autovectores[:, 1]  # Segundo autovector

#La matriz de transformación será la misma matriz de los autovectores
T = autovectores

#---------------------------------------------------------------------------

#PUNTO B CASO CON LAS 3 COORDENADAS

#PUNTO B.1 MOMENTO DE INERCIA ORNDEN 0 ES EL MISMO QUE EL DEL CASO 2D:
momentocero_2=momentocero

#PUNTO B.2 MOMENTO DE INERCIA ORDEN 1

# Extraer la nueva columna de datos:
z = df['z']

# Calcular centro de masa(x_cm y y_cm seran iguales al caso 2D)
z_cm = (m * z).sum() / masa_total

#PUNTO B.3 MOMENTO DE INERCIA ORDEN 2
# Calcular momentos de inercia:
# Coordenadas relativas al centro de masa(x_rel y y_rel son iguales al caso 2D)
z_rel = z - z_cm

#Calculamos las componentes del tensor de incercia:
I_xx = (m * (y_rel**2 + z_rel**2)).sum()
I_yy = (m * (x_rel**2 + z_rel**2)).sum()
I_zz = (m * (x_rel**2 + y_rel**2)).sum()

I_xy = -(m * x_rel * y_rel).sum()
I_xz = -(m * x_rel * z_rel).sum()
I_yz = -(m * y_rel * z_rel).sum()

I_tensor_2 = np.array([[I_xx, I_xy, I_xz],
                     [I_xy, I_yy, I_yz],
                     [I_xz, I_yz, I_zz]])

#Calculamos los autovalores y autovectores:
autovalores_2, autovectores_2 = np.linalg.eigh(I_tensor_2)
#Autovectores desde la matriz (columnas)
Eje_prin_1 = autovectores_2[:, 0]  # Primer autovector
Eje_prin_2= autovectores_2[:, 1]  # Segundo autovector
Eje_prin_3= autovectores_2[:, 2] # Tercer autovector

#Matriz de transformación:
T_2=autovectores_2

#--------------------------------------------------------------------------
#GRÁFICOS:

masas = df['masas']

eje1 = autovectores_2[:, 0] * autovalores_2[0]
eje2 = autovectores_2[:, 1] * autovalores_2[1]
eje3 = autovectores_2[:, 2] * autovalores_2[2]

# --- Gráfico 1: Partículas y CM ---
plt.figure(figsize=(15, 5))

# 2D
plt.subplot(1, 2, 1)
plt.scatter(x, y, s=masas*20, alpha=0.7, color='orange', label='Partículas')
plt.scatter(x_cm, y_cm, c='red', s=100, marker='*', label='Centro de masa')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Partículas en 2D (XY)')
plt.legend()
plt.grid()

# 3D
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(x, y, z, s=masas*20, alpha=0.7, color='orange', label='Partículas')
ax.scatter(x_cm, y_cm, z_cm, c='red', s=100, marker='*', label='Centro de masa')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Partículas en 3D')
ax.legend()

plt.tight_layout()
plt.savefig("particulas_cm.png")
plt.show()

# --- Gráfico 2: Ejes principales de inercia ---
plt.figure(figsize=(15, 5))

# 2D con ejes principales
plt.subplot(1, 2, 1)
plt.scatter(x_rel, y_rel, s=masas*20, alpha=0.7, color='orange', label='Partículas (rel. CM)')
plt.quiver(0, 0, eje1[0], eje1[1], color='cyan', scale=1, scale_units='xy', angles='xy', label='Eje principal 1')
plt.quiver(0, 0, eje2[0], eje2[1], color='crimson', scale=1, scale_units='xy', angles='xy', label='Eje principal 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ejes principales de inercia (XY)')
plt.legend()
plt.grid()

# 3D con ejes principales
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(x_rel, y_rel, z_rel, s=masas*20, alpha=0.7, color='orange', label='Partículas (rel. CM)')
ax.quiver(0, 0, 0, eje1[0], eje1[1], eje1[2], color='cyan', label='Eje 1')
ax.quiver(0, 0, 0, eje2[0], eje2[1], eje2[2], color='crimson', label='Eje 2')
ax.quiver(0, 0, 0, eje3[0], eje3[1], eje3[2], color='lime', label='Eje 3')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ejes principales de inercia (3D)')
ax.legend()

plt.tight_layout()
plt.savefig("ejes_principales_inercia.png")
plt.show()

#--------------------------------------------------------------------------

print("RESPUESTAS CASO 2D:")

print("1) Momento de orden cero: ", momentocero)

print(f"2) Momento de orden uno(Posición del centro de masa):({x_cm:.4f}, {y_cm:.4f}) ")

print("3) Tensor de inercia (2D) respecto al centro de masa(Momento de orden dos): ")
print(I_tensor)

print("4) Ejes principales: ")
print("Autovector 1:",Eje_principal_1)
print("Autovector 2:", Eje_principal_2)

print("5) Matriz de transformación:",T)
print("------------------------------------------------")
print("RESPUESTAS CASO 3D:")
print("1) Momento de orden cero: ", momentocero_2)
print("2) Momento de orden uno(Posición del centro de masa):({:.4f}, {:.4f}, {:.4f}) ".format(x_cm, y_cm, z_cm))
print("3) Tensor de inercia (3D) respecto al centro de masa(Momento de orden dos): ")
print(I_tensor_2)
print("4) Ejes principales: ")
print("Autovector 1:", Eje_prin_1)
print("Autovector 2:", Eje_prin_2)
print("Autovector 3:", Eje_prin_3)
print("5) Matriz de transformación:",T_2)
