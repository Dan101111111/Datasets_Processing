# Ejercicio 3: Preprocesamiento del Dataset Iris
# Objetivo: Flujo completo de preprocesamiento y visualización de resultados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset desde sklearn.datasets
print("=== CARGA DEL DATASET ===")
iris = load_iris()

print(f"Shape: {iris.data.shape}")
print(f"Número de clases: {len(iris.target_names)}")
print(f"Clases: {iris.target_names}")

# 2. Convertir a DataFrame y agregar nombres de columnas
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
dataset['target'] = iris.target

print(f"\nPrimeras 5 filas del dataset:")
print(dataset.head())

print(f"\nInformación del dataset:")
print(dataset.info())

# 3. Separar características (X) y variable objetivo (y)
X = dataset.drop('target', axis=1)
y = dataset['target'].values

# 4. Aplicar estandarización con StandardScaler
# Esto centra los datos en media=0 y desviación estándar=1
print("\n=== ESTANDARIZACIÓN ===")
print("Estadísticas ANTES de estandarizar:")
print(X.describe())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir a DataFrame para mantener nombres de columnas
X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names)

print("\nEstadísticas DESPUÉS de estandarizar:")
print(X_scaled_df.describe())

# 5. Dividir dataset (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42
)

print("\n=== DIMENSIONES ===")
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")

# 6. Gráfico de dispersión: sepal length vs petal length diferenciado por clase
print("\n=== GENERANDO GRÁFICO ===")

plt.figure(figsize=(10, 6))

# Crear gráfico para cada clase con diferentes colores
colores = ['red', 'green', 'blue']
nombres_clases = iris.target_names

for i, (color, nombre) in enumerate(zip(colores, nombres_clases)):
    # Filtrar datos por clase
    indices = y == i
    plt.scatter(
        X_scaled_df.loc[indices, 'sepal length (cm)'],
        X_scaled_df.loc[indices, 'petal length (cm)'],
        c=color,
        label=nombre,
        alpha=0.6,
        edgecolors='black',
        s=80
    )

plt.xlabel('Sepal Length (cm) - Estandarizado', fontsize=12)
plt.ylabel('Petal Length (cm) - Estandarizado', fontsize=12)
plt.title('Distribución Sepal Length vs Petal Length por Clase', fontsize=14, fontweight='bold')
plt.legend(title='Clase')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Mostrar gráfico
plt.show()

# SALIDA ESPERADA: Estadísticas descriptivas del dataset estandarizado
print("\n=== ESTADÍSTICAS DESCRIPTIVAS (ESTANDARIZADO) ===")
print(X_scaled_df.describe())

print("\n=== INTERPRETACIÓN ===")
print("Medias cercanas a 0 y desviaciones estándar cercanas a 1")
print("Esto confirma que la estandarización fue exitosa")