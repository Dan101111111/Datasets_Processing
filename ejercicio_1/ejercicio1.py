# Ejercicio 1: Preprocesamiento del Dataset Titanic
# Objetivo: Preparar los datos para predecir la supervivencia de los pasajeros

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset
dataset = pd.read_csv("Titanic-Dataset.csv")

# 2. Eliminar columnas irrelevantes
dataset = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 3. Separar variables independientes (X) y dependiente (y)
y = dataset['Survived'].values
X = dataset.drop('Survived', axis=1)

# 4. Verificar y reemplazar valores nulos
# Identificar columnas numéricas y categóricas
columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = X.select_dtypes(include=['object']).columns.tolist()

# Imputar nulos: media para numéricas, moda para categóricas
# Esto evita errores en los modelos de ML
imputer_numeric = SimpleImputer(missing_values=np.nan, strategy="mean")
X[columnas_numericas] = imputer_numeric.fit_transform(X[columnas_numericas])

imputer_categorical = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
X[columnas_categoricas] = imputer_categorical.fit_transform(X[columnas_categoricas].values.reshape(-1, len(columnas_categoricas)))

# 5. Codificar variables Sex y Embarked
# Convertir texto a números: Sex (female=0, male=1), Embarked (C=0, Q=1, S=2)
le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])

le_embarked = LabelEncoder()
X['Embarked'] = le_embarked.fit_transform(X['Embarked'])

# 6. Dividir datos en entrenamiento (70%) y prueba (30%)
# Importante: dividir antes de escalar para evitar data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 7. Estandarizar variables numéricas (Age, Fare)
# Fórmula: (x - media) / desviación estándar -> resultado con media≈0 y std≈1
scaler = StandardScaler()
columnas_a_escalar = ['Age', 'Fare']
indices_columnas = [X_train.columns.get_loc(col) for col in columnas_a_escalar]

X_train_array = X_train.values
X_test_array = X_test.values

# fit_transform en train (aprende parámetros), solo transform en test
X_train_array[:, indices_columnas] = scaler.fit_transform(X_train_array[:, indices_columnas])
X_test_array[:, indices_columnas] = scaler.transform(X_test_array[:, indices_columnas])

X_train = pd.DataFrame(X_train_array, columns=X_train.columns)
X_test = pd.DataFrame(X_test_array, columns=X_test.columns)

# SALIDA ESPERADA
print("\n=== PRIMEROS 5 REGISTROS PROCESADOS ===")
print(X_train.head())

print("\n=== DIMENSIONES ===")
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")