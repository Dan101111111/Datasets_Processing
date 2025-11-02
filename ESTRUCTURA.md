# 📁 Estructura del Proyecto

```
datasets_processing/
│
├── 📄 app.py                          # ⭐ Aplicación principal de Streamlit
├── 📄 requirements.txt                # 📦 Dependencias del proyecto
├── 📄 README.md                       # 📖 Documentación principal
├── 📄 DEPLOYMENT.md                   # 🚀 Guía de deployment
├── 📄 GIT_COMMANDS.md                 # 🔧 Comandos Git útiles
├── 📄 run.bat                         # 🎯 Script de inicio rápido (Windows)
├── 📄 .gitignore                      # 🚫 Archivos a ignorar en Git
│
├── 📁 .streamlit/
│   └── 📄 config.toml                 # ⚙️ Configuración de Streamlit
│
├── 📁 ejercicio_1/
│   ├── 📄 ejercicio1.py               # 🐍 Script original
│   └── 📊 Titanic-Dataset.csv         # 📊 Dataset del Titanic
│
├── 📁 ejercicio_2/
│   ├── 📄 ejercicio2.py               # 🐍 Script original
│   └── 📊 student-mat.csv             # 📊 Dataset de estudiantes
│
└── 📁 ejercicio_3/
    └── 📄 ejercicio3.py               # 🐍 Script original (usa Iris de sklearn)
```

## 🎯 Archivos Principales

### `app.py`

La aplicación principal de Streamlit que integra los 3 ejercicios con:

- Navegación por sidebar
- Visualizaciones interactivas
- Procesamiento paso a paso
- Análisis estadístico detallado

### `requirements.txt`

Lista todas las dependencias necesarias:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### `README.md`

Documentación completa con:

- Descripción del proyecto
- Instrucciones de instalación
- Guía de uso
- Características de cada ejercicio

### `DEPLOYMENT.md`

Guía paso a paso para hacer deploy en Streamlit Cloud:

- Preparación del repositorio
- Configuración en Streamlit Cloud
- Troubleshooting
- Tips de optimización

### `run.bat`

Script de Windows para inicio rápido:

- Crea entorno virtual automáticamente
- Instala dependencias
- Ejecuta la aplicación

## 📊 Datasets Incluidos

### 1. Titanic-Dataset.csv (891 registros)

- PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

### 2. student-mat.csv (395 registros)

- 33 variables sobre estudiantes de matemáticas
- Incluye: school, sex, age, address, famsize, etc.
- Notas: G1, G2, G3

### 3. Iris Dataset

- Cargado desde sklearn.datasets
- 150 muestras, 4 características
- 3 especies de flores

## 🔄 Flujo de Trabajo

```
1. Usuario ejecuta run.bat (o streamlit run app.py)
   ↓
2. Streamlit inicia el servidor
   ↓
3. Se abre el navegador en localhost:8501
   ↓
4. Usuario navega entre ejercicios
   ↓
5. App carga y procesa datos en tiempo real
   ↓
6. Muestra visualizaciones y resultados
```

## 🚀 Para Deploy en Streamlit Cloud

```
1. Subir proyecto a GitHub
   ↓
2. Conectar en share.streamlit.io
   ↓
3. Seleccionar repositorio y app.py
   ↓
4. Deploy automático
   ↓
5. App disponible públicamente
```

## 📝 Notas Importantes

- ✅ Todos los archivos CSV deben estar en sus carpetas respectivas
- ✅ No incluir entornos virtuales en Git (ya está en .gitignore)
- ✅ Las rutas son relativas, funcionan en local y en cloud
- ✅ El dataset Iris no requiere archivo, se carga de sklearn

## 🎨 Características de la App

### Página de Inicio

- Descripción del proyecto
- Navegación a cada ejercicio
- Tecnologías utilizadas

### Ejercicio 1: Titanic

- Métricas principales
- Proceso paso a paso
- Visualización de supervivencia
- Gráficos por género

### Ejercicio 2: Students

- Estadísticas de estudiantes
- One-Hot Encoding visual
- Matriz de correlación
- Distribución de notas

### Ejercicio 3: Iris

- Carga desde sklearn
- Comparación antes/después estandarización
- Scatter plots por especie
- Matriz de correlación

## 🛠️ Personalización

Para personalizar colores y tema, edita `.streamlit/config.toml`:

- primaryColor: Color principal de botones
- backgroundColor: Color de fondo
- secondaryBackgroundColor: Color de fondo secundario
- textColor: Color del texto

## 📊 Uso de Memoria

Aproximado por ejercicio:

- Titanic: ~100 KB
- Students: ~50 KB
- Iris: ~10 KB (en memoria)

Total: < 200 KB - Perfecto para Streamlit Cloud Free Tier

## 🔐 Seguridad

- ✅ No incluye datos sensibles
- ✅ .gitignore configurado correctamente
- ✅ Sin contraseñas o API keys hardcoded
- ✅ Listo para repositorio público

---

**¡Todo listo para visualizar y hacer deploy! 🎉**
