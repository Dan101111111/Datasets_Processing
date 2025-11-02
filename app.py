"""
Aplicación Streamlit - Preprocesamiento de Datasets
Proyecto de Sistemas Inteligentes - Procesamiento de Datos
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import os

# Configuración de la página
st.set_page_config(
    page_title="Preprocesamiento de Datasets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Navegación
st.sidebar.title("📊 Navegación")
st.sidebar.markdown("---")
ejercicio = st.sidebar.radio(
    "Selecciona un ejercicio:",
    ["🏠 Inicio", "🚢 Ejercicio 1: Titanic", "📚 Ejercicio 2: Student Performance", "🌸 Ejercicio 3: Iris Dataset"]
)

# Función para el Ejercicio 1 - Titanic
def ejercicio_titanic():
    st.markdown("<h1 class='main-header'>🚢 Ejercicio 1: Dataset Titanic</h1>", unsafe_allow_html=True)
    st.markdown("### Objetivo: Preparar los datos para predecir la supervivencia de los pasajeros")
    
    # Verificar que el archivo existe
    file_path = os.path.join("ejercicio_1", "Titanic-Dataset.csv")
    if not os.path.exists(file_path):
        st.error(f"⚠️ No se encontró el archivo: {file_path}")
        return
    
    # Cargar dataset
    dataset = pd.read_csv(file_path)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", dataset.shape[0])
    with col2:
        st.metric("Total de Variables", dataset.shape[1])
    with col3:
        st.metric("Supervivientes", dataset['Survived'].sum())
    
    # Mostrar dataset original
    st.markdown("<h2 class='section-header'>📋 Dataset Original</h2>", unsafe_allow_html=True)
    st.dataframe(dataset.head(10), use_container_width=True)
    
    # Información del dataset
    with st.expander("ℹ️ Información del Dataset"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Tipos de datos:**")
            st.dataframe(dataset.dtypes.to_frame('Tipo'), use_container_width=True)
        with col2:
            st.write("**Valores nulos:**")
            st.dataframe(dataset.isnull().sum().to_frame('Nulos'), use_container_width=True)
    
    st.markdown("<h2 class='section-header'>🔧 Proceso de Preprocesamiento</h2>", unsafe_allow_html=True)
    
    # Paso 1: Eliminar columnas irrelevantes
    st.write("**1. Eliminando columnas irrelevantes:** PassengerId, Name, Ticket, Cabin")
    dataset_clean = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Paso 2: Separar X e y
    st.write("**2. Separando variable objetivo (Survived) de las características**")
    y = dataset_clean['Survived'].values
    X = dataset_clean.drop('Survived', axis=1)
    
    # Paso 3: Imputar valores nulos
    st.write("**3. Imputando valores nulos:**")
    col1, col2 = st.columns(2)
    with col1:
        st.info("📊 Variables numéricas: Media")
    with col2:
        st.info("📝 Variables categóricas: Moda")
    
    columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
    
    imputer_numeric = SimpleImputer(missing_values=np.nan, strategy="mean")
    X[columnas_numericas] = imputer_numeric.fit_transform(X[columnas_numericas])
    
    imputer_categorical = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    X[columnas_categoricas] = imputer_categorical.fit_transform(X[columnas_categoricas].values.reshape(-1, len(columnas_categoricas)))
    
    # Paso 4: Codificar variables categóricas
    st.write("**4. Codificando variables categóricas (Sex, Embarked)**")
    le_sex = LabelEncoder()
    X['Sex'] = le_sex.fit_transform(X['Sex'])
    
    le_embarked = LabelEncoder()
    X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
    
    # Paso 5: División train/test
    st.write("**5. Dividiendo datos: 70% entrenamiento, 30% prueba**")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # Paso 6: Estandarización
    st.write("**6. Estandarizando variables numéricas (Age, Fare)**")
    scaler = StandardScaler()
    columnas_a_escalar = ['Age', 'Fare']
    indices_columnas = [X_train.columns.get_loc(col) for col in columnas_a_escalar]
    
    X_train_array = X_train.values
    X_test_array = X_test.values
    
    X_train_array[:, indices_columnas] = scaler.fit_transform(X_train_array[:, indices_columnas])
    X_test_array[:, indices_columnas] = scaler.transform(X_test_array[:, indices_columnas])
    
    X_train_final = pd.DataFrame(X_train_array, columns=X_train.columns)
    X_test_final = pd.DataFrame(X_test_array, columns=X_test.columns)
    
    # Resultados
    st.markdown("<h2 class='section-header'>✅ Resultados del Preprocesamiento</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("X_train", f"{X_train_final.shape[0]} × {X_train_final.shape[1]}")
    with col2:
        st.metric("X_test", f"{X_test_final.shape[0]} × {X_test_final.shape[1]}")
    with col3:
        st.metric("y_train", len(y_train))
    with col4:
        st.metric("y_test", len(y_test))
    
    st.write("**Primeros 5 registros procesados:**")
    st.dataframe(X_train_final.head(), use_container_width=True)
    
    # Visualización
    st.markdown("<h2 class='section-header'>📊 Visualización de Datos</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        survived_counts = pd.Series(y).value_counts()
        colors = ['#ff6b6b', '#51cf66']
        ax.pie(survived_counts, labels=['No Sobrevivió', 'Sobrevivió'], autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('Distribución de Supervivencia', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        dataset_clean.groupby(['Sex', 'Survived']).size().unstack().plot(kind='bar', ax=ax, color=['#ff6b6b', '#51cf66'])
        ax.set_title('Supervivencia por Sexo', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sexo (0: Female, 1: Male)')
        ax.set_ylabel('Cantidad')
        ax.legend(['No Sobrevivió', 'Sobrevivió'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)


# Función para el Ejercicio 2 - Student Performance
def ejercicio_student():
    st.markdown("<h1 class='main-header'>📚 Ejercicio 2: Student Performance</h1>", unsafe_allow_html=True)
    st.markdown("### Objetivo: Procesar datos para predecir la nota final (G3) de los estudiantes")
    
    file_path = os.path.join("ejercicio_2", "student-mat.csv")
    if not os.path.exists(file_path):
        st.error(f"⚠️ No se encontró el archivo: {file_path}")
        return
    
    dataset = pd.read_csv(file_path)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Estudiantes", dataset.shape[0])
    with col2:
        st.metric("Total de Variables", dataset.shape[1])
    with col3:
        st.metric("Nota Promedio (G3)", f"{dataset['G3'].mean():.2f}")
    
    st.markdown("<h2 class='section-header'>📋 Dataset Original</h2>", unsafe_allow_html=True)
    st.dataframe(dataset.head(10), use_container_width=True)
    
    # Análisis inicial
    with st.expander("ℹ️ Análisis Inicial"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Variables Categóricas:**")
            categoricas = dataset.select_dtypes(include=['object']).columns.tolist()
            st.write(categoricas)
        with col2:
            st.write("**Variables Numéricas:**")
            numericas = dataset.select_dtypes(include=['int64']).columns.tolist()
            st.write(numericas)
    
    st.markdown("<h2 class='section-header'>🔧 Proceso de Preprocesamiento</h2>", unsafe_allow_html=True)
    
    # Paso 1: Limpieza
    duplicados = dataset.duplicated().sum()
    st.write(f"**1. Limpieza de datos:** {duplicados} duplicados encontrados y eliminados")
    dataset_clean = dataset.drop_duplicates()
    
    # Paso 2: Separar variable objetivo
    st.write("**2. Separando variable objetivo (G3)**")
    y = dataset_clean['G3'].values
    X = dataset_clean.drop('G3', axis=1)
    
    # Paso 3: One Hot Encoding
    st.write("**3. Aplicando One Hot Encoding a variables categóricas**")
    categoricas = X.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 Columnas antes: {X.shape[1]}")
    
    X_encoded = pd.get_dummies(X, columns=categoricas, drop_first=True)
    
    with col2:
        st.info(f"📊 Columnas después: {X_encoded.shape[1]}")
    
    # Paso 4: Normalización
    st.write("**4. Normalizando variables numéricas (age, absences, G1, G2) al rango [0, 1]**")
    columnas_a_normalizar = ['age', 'absences', 'G1', 'G2']
    
    scaler = MinMaxScaler()
    X_encoded[columnas_a_normalizar] = scaler.fit_transform(X_encoded[columnas_a_normalizar])
    
    # Paso 5: División train/test
    st.write("**5. Dividiendo datos: 80% entrenamiento, 20% prueba**")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42
    )
    
    # Resultados
    st.markdown("<h2 class='section-header'>✅ Resultados del Preprocesamiento</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("X_train", f"{X_train.shape[0]} × {X_train.shape[1]}")
    with col2:
        st.metric("X_test", f"{X_test.shape[0]} × {X_test.shape[1]}")
    with col3:
        st.metric("y_train", len(y_train))
    with col4:
        st.metric("y_test", len(y_test))
    
    st.write("**Primeros 5 registros procesados:**")
    st.dataframe(X_train.head(), use_container_width=True)
    
    # Análisis de correlación
    st.markdown("<h2 class='section-header'>📊 Análisis de Correlación entre Notas</h2>", unsafe_allow_html=True)
    
    notas = dataset_clean[['G1', 'G2', 'G3']]
    correlacion = notas.corr()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Matriz de Correlación:**")
        st.dataframe(correlacion.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1), use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlacion, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Correlación entre Notas G1, G2 y G3', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    st.info(f"""
    **Interpretación:**
    - Correlación G1-G2: {correlacion.loc['G1', 'G2']:.3f}
    - Correlación G1-G3: {correlacion.loc['G1', 'G3']:.3f}
    - Correlación G2-G3: {correlacion.loc['G2', 'G3']:.3f}
    
    Valores cercanos a 1 indican alta correlación positiva
    """)
    
    # Visualizaciones adicionales
    st.markdown("<h2 class='section-header'>📊 Visualizaciones Adicionales</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(dataset_clean['G3'], bins=20, color='#4CAF50', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Nota Final (G3)', fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title('Distribución de Notas Finales', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(dataset_clean['G2'], dataset_clean['G3'], alpha=0.6, c='#2196F3', edgecolors='black')
        ax.set_xlabel('Nota G2', fontsize=12)
        ax.set_ylabel('Nota G3 (Final)', fontsize=12)
        ax.set_title('Relación entre G2 y G3', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


# Función para el Ejercicio 3 - Iris
def ejercicio_iris():
    st.markdown("<h1 class='main-header'>🌸 Ejercicio 3: Iris Dataset</h1>", unsafe_allow_html=True)
    st.markdown("### Objetivo: Flujo completo de preprocesamiento y visualización")
    
    # Cargar dataset
    iris = load_iris()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Muestras", iris.data.shape[0])
    with col2:
        st.metric("Características", iris.data.shape[1])
    with col3:
        st.metric("Clases", len(iris.target_names))
    
    # Convertir a DataFrame
    dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataset['target'] = iris.target
    dataset['species'] = dataset['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    
    st.markdown("<h2 class='section-header'>📋 Dataset Original</h2>", unsafe_allow_html=True)
    st.dataframe(dataset.head(10), use_container_width=True)
    
    with st.expander("ℹ️ Información del Dataset"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Clases:**")
            for i, name in enumerate(iris.target_names):
                count = (dataset['target'] == i).sum()
                st.write(f"- {name}: {count} muestras")
        with col2:
            st.write("**Estadísticas descriptivas:**")
            st.dataframe(dataset.describe(), use_container_width=True)
    
    st.markdown("<h2 class='section-header'>🔧 Proceso de Preprocesamiento</h2>", unsafe_allow_html=True)
    
    # Separar X e y
    st.write("**1. Separando características (X) y variable objetivo (y)**")
    X = dataset.drop(['target', 'species'], axis=1)
    y = dataset['target'].values
    
    # Estadísticas antes de estandarizar
    st.write("**2. Estadísticas ANTES de estandarizar:**")
    st.dataframe(X.describe(), use_container_width=True)
    
    # Estandarización
    st.write("**3. Aplicando estandarización (StandardScaler)**")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
    
    st.write("**4. Estadísticas DESPUÉS de estandarizar:**")
    st.dataframe(X_scaled_df.describe(), use_container_width=True)
    
    st.success("✅ Medias cercanas a 0 y desviaciones estándar cercanas a 1 - Estandarización exitosa!")
    
    # División train/test
    st.write("**5. Dividiendo datos: 70% entrenamiento, 30% prueba**")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.30, random_state=42
    )
    
    # Resultados
    st.markdown("<h2 class='section-header'>✅ Resultados del Preprocesamiento</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("X_train", f"{X_train.shape[0]} × {X_train.shape[1]}")
    with col2:
        st.metric("X_test", f"{X_test.shape[0]} × {X_test.shape[1]}")
    with col3:
        st.metric("y_train", len(y_train))
    with col4:
        st.metric("y_test", len(y_test))
    
    # Visualizaciones
    st.markdown("<h2 class='section-header'>📊 Visualizaciones</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 8))
        colores = ['red', 'green', 'blue']
        nombres_clases = iris.target_names
        
        for i, (color, nombre) in enumerate(zip(colores, nombres_clases)):
            indices = y == i
            ax.scatter(
                X_scaled_df.loc[indices, 'sepal length (cm)'],
                X_scaled_df.loc[indices, 'petal length (cm)'],
                c=color,
                label=nombre,
                alpha=0.6,
                edgecolors='black',
                s=80
            )
        
        ax.set_xlabel('Sepal Length (cm) - Estandarizado', fontsize=12)
        ax.set_ylabel('Petal Length (cm) - Estandarizado', fontsize=12)
        ax.set_title('Sepal Length vs Petal Length', fontsize=14, fontweight='bold')
        ax.legend(title='Especie')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, (color, nombre) in enumerate(zip(colores, nombres_clases)):
            indices = y == i
            ax.scatter(
                X_scaled_df.loc[indices, 'sepal width (cm)'],
                X_scaled_df.loc[indices, 'petal width (cm)'],
                c=color,
                label=nombre,
                alpha=0.6,
                edgecolors='black',
                s=80
            )
        
        ax.set_xlabel('Sepal Width (cm) - Estandarizado', fontsize=12)
        ax.set_ylabel('Petal Width (cm) - Estandarizado', fontsize=12)
        ax.set_title('Sepal Width vs Petal Width', fontsize=14, fontweight='bold')
        ax.legend(title='Especie')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Matriz de correlación
    st.markdown("<h2 class='section-header'>🔗 Matriz de Correlación</h2>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    correlacion = X.corr()
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Correlación entre Características', fontsize=14, fontweight='bold')
    st.pyplot(fig)


# Página de inicio
def pagina_inicio():
    st.markdown("<h1 class='main-header'>📊 Preprocesamiento de Datasets</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>Proyecto de Sistemas Inteligentes - Procesamiento de Datos</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Bienvenido
    
    Esta aplicación presenta tres ejercicios completos de preprocesamiento de datos para Machine Learning.
    Utiliza el menú lateral para navegar entre los diferentes ejercicios.
    
    ### 📚 Contenido del Proyecto:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h3>🚢 Ejercicio 1: Titanic</h3>
        <ul>
            <li>Limpieza de datos</li>
            <li>Imputación de valores nulos</li>
            <li>Codificación de variables</li>
            <li>Estandarización</li>
            <li>División train/test</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
        <h3>📚 Ejercicio 2: Students</h3>
        <ul>
            <li>Eliminación de duplicados</li>
            <li>One-Hot Encoding</li>
            <li>Normalización MinMax</li>
            <li>Análisis de correlación</li>
            <li>Visualizaciones</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-box'>
        <h3>🌸 Ejercicio 3: Iris</h3>
        <ul>
            <li>Carga desde sklearn</li>
            <li>Estandarización completa</li>
            <li>Análisis estadístico</li>
            <li>Visualizaciones avanzadas</li>
            <li>Matriz de correlación</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🛠️ Tecnologías Utilizadas:
    
    - **Python**: Lenguaje de programación principal
    - **Pandas**: Manipulación y análisis de datos
    - **NumPy**: Operaciones numéricas
    - **Scikit-learn**: Preprocesamiento y machine learning
    - **Matplotlib & Seaborn**: Visualización de datos
    - **Streamlit**: Framework para la aplicación web
    
    ### 📖 Instrucciones:
    
    1. Selecciona un ejercicio desde el menú lateral
    2. Explora el proceso de preprocesamiento paso a paso
    3. Analiza las visualizaciones y resultados
    4. Revisa las métricas y estadísticas generadas
    
    ---
    
    <p style='text-align: center; color: #999; margin-top: 2rem;'>
    Desarrollado por Ordóñez Ugarte, Jesús Daniel
    </p>
    """, unsafe_allow_html=True)


# Navegación principal
if ejercicio == "🏠 Inicio":
    pagina_inicio()
elif ejercicio == "🚢 Ejercicio 1: Titanic":
    ejercicio_titanic()
elif ejercicio == "📚 Ejercicio 2: Student Performance":
    ejercicio_student()
elif ejercicio == "🌸 Ejercicio 3: Iris Dataset":
    ejercicio_iris()

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #999; font-size: 0.9rem;'>
© 2025 - Preprocesamiento de Datasets | Sistemas Inteligentes
</p>
""", unsafe_allow_html=True)
