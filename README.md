# 📊 Preprocesamiento de Datasets - Streamlit App

Aplicación web interactiva desarrollada con Streamlit que presenta tres ejercicios completos de preprocesamiento de datos para Machine Learning.

## 🎯 Descripción del Proyecto

Este proyecto contiene tres ejercicios prácticos que demuestran diferentes técnicas de preprocesamiento de datos:

1. **🚢 Ejercicio 1: Dataset Titanic** - Predicción de supervivencia
2. **📚 Ejercicio 2: Student Performance** - Predicción de notas finales
3. **🌸 Ejercicio 3: Iris Dataset** - Clasificación de especies de flores

## 🚀 Instalación Local

### Prerequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el proyecto**

```bash
cd datasets_processing
```

2. **Crear un entorno virtual (recomendado)**

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar las dependencias**

```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📦 Estructura del Proyecto

```
datasets_processing/
│
├── app.py                          # Aplicación principal de Streamlit
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Este archivo
│
├── ejercicio_1/
│   ├── ejercicio1.py              # Script original del ejercicio 1
│   └── Titanic-Dataset.csv        # Dataset del Titanic
│
├── ejercicio_2/
│   ├── ejercicio2.py              # Script original del ejercicio 2
│   └── student-mat.csv            # Dataset de rendimiento estudiantil
│
└── ejercicio_3/
    └── ejercicio3.py              # Script original del ejercicio 3
```

## 🌐 Deployment en Streamlit Cloud

### Opción 1: Deployment desde GitHub

1. **Subir el proyecto a GitHub**

```bash
# Inicializar repositorio git (si no existe)
git init

# Agregar archivos
git add .

# Hacer commit
git commit -m "Initial commit - Datasets Processing App"

# Conectar con tu repositorio remoto
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git

# Subir los cambios
git push -u origin main
```

2. **Deployment en Streamlit Cloud**

   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Inicia sesión con tu cuenta de GitHub
   - Click en "New app"
   - Selecciona tu repositorio
   - Configuración:
     - **Main file path**: `app.py`
     - **Python version**: 3.9 o superior
   - Click en "Deploy"

### Opción 2: Configuración Avanzada

Si necesitas configuraciones adicionales, puedes crear archivos:

**`.streamlit/config.toml`** (opcional):

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

**`.gitignore`**:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Streamlit
.streamlit/secrets.toml

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## 📊 Características de la Aplicación

### Ejercicio 1: Dataset Titanic

- ✅ Limpieza de datos (eliminación de columnas irrelevantes)
- ✅ Imputación de valores nulos (media y moda)
- ✅ Codificación de variables categóricas (LabelEncoder)
- ✅ Estandarización de variables numéricas (StandardScaler)
- ✅ División train/test (70/30)
- ✅ Visualizaciones de supervivencia

### Ejercicio 2: Student Performance

- ✅ Eliminación de duplicados
- ✅ One-Hot Encoding para variables categóricas
- ✅ Normalización MinMax (rango 0-1)
- ✅ Análisis de correlación entre notas
- ✅ División train/test (80/20)
- ✅ Visualizaciones de distribución y correlación

### Ejercicio 3: Iris Dataset

- ✅ Carga desde sklearn.datasets
- ✅ Estandarización completa (StandardScaler)
- ✅ Análisis estadístico detallado
- ✅ División train/test (70/30)
- ✅ Visualizaciones de dispersión por clase
- ✅ Matriz de correlación entre características

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework para aplicaciones web de datos
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Computación numérica
- **Scikit-learn**: Preprocesamiento y machine learning
- **Matplotlib**: Visualización de datos
- **Seaborn**: Visualización estadística

## 📝 Notas Importantes

1. **Archivos de datos**: Asegúrate de que los archivos CSV estén en sus respectivas carpetas:

   - `ejercicio_1/Titanic-Dataset.csv`
   - `ejercicio_2/student-mat.csv`

2. **Dataset Iris**: Se carga automáticamente desde scikit-learn, no requiere archivo CSV.

3. **Memoria**: La aplicación carga los datasets en memoria, ideal para datasets pequeños y medianos.

## 🐛 Solución de Problemas

### Error: "No se encontró el archivo CSV"

- Verifica que los archivos CSV estén en las carpetas correctas
- Asegúrate de ejecutar la app desde el directorio raíz del proyecto

### Error de dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Puerto ya en uso

```bash
streamlit run app.py --server.port 8502
```
