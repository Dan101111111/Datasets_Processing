# Ejercicio 2: Procesamiento del Dataset Student Performance
# Objetivo: Procesar los datos para predecir la nota final (G3) de los estudiantes

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset y analizar variables categóricas
dataset = pd.read_csv("student-mat.csv")

print("=== ANÁLISIS INICIAL ===")
print(f"Shape del dataset: {dataset.shape}")
print(f"\nVariables categóricas:")
categoricas = dataset.select_dtypes(include=['object']).columns.tolist()
print(categoricas)
print(f"\nVariables numéricas:")
numericas = dataset.select_dtypes(include=['int64']).columns.tolist()
print(numericas)

# 2. Eliminar duplicados y valores inconsistentes
print(f"\n=== LIMPIEZA DE DATOS ===")
print(f"Duplicados encontrados: {dataset.duplicated().sum()}")
dataset = dataset.drop_duplicates()

# Verificar valores nulos
print(f"Valores nulos: {dataset.isnull().sum().sum()}")

# 3. Separar variable objetivo (G3) antes del encoding
y = dataset['G3'].values
X = dataset.drop('G3', axis=1)

# 4. Aplicar One Hot Encoding a variables categóricas
# Esto convierte cada categoría en una columna binaria (0 o 1)
print(f"\n=== ONE HOT ENCODING ===")
print(f"Columnas antes del encoding: {X.shape[1]}")

X_encoded = pd.get_dummies(X, columns=categoricas, drop_first=True)

print(f"Columnas después del encoding: {X_encoded.shape[1]}")
print(f"Nuevas columnas creadas: {X_encoded.shape[1] - X.shape[1]}")

# 5. Normalizar variables numéricas específicas (age, absences, G1, G2)
# La normalización escala los valores al rango [0, 1]
print(f"\n=== NORMALIZACIÓN ===")
columnas_a_normalizar = ['age', 'absences', 'G1', 'G2']

scaler = MinMaxScaler()
X_encoded[columnas_a_normalizar] = scaler.fit_transform(X_encoded[columnas_a_normalizar])

print(f"Variables normalizadas: {columnas_a_normalizar}")
print(f"Rango de valores: [0, 1]")

# 6. Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42
)

# SALIDA ESPERADA
print("\n=== PRIMEROS 5 REGISTROS PROCESADOS ===")
print(X_train.head())

print("\n=== DIMENSIONES ===")
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")

# RETO ADICIONAL: Análisis de correlación entre G1, G2 y G3
print("\n=== RETO ADICIONAL: CORRELACIÓN ENTRE NOTAS ===")
notas = dataset[['G1', 'G2', 'G3']]
correlacion = notas.corr()
print("\nMatriz de correlación:")
print(correlacion)

print("\nInterpretación:")
print(f"Correlación G1-G2: {correlacion.loc['G1', 'G2']:.3f}")
print(f"Correlación G1-G3: {correlacion.loc['G1', 'G3']:.3f}")
print(f"Correlación G2-G3: {correlacion.loc['G2', 'G3']:.3f}")
print("\nNota: Valores cercanos a 1 indican alta correlación positiva")