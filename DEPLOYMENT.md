# Guía Rápida de Deployment en Streamlit Cloud

## 🚀 Pasos para Deploy

### 1. Preparar el Repositorio en GitHub

```bash
# Inicializar Git (si no está inicializado)
git init

# Agregar todos los archivos
git add .

# Hacer el primer commit
git commit -m "Initial commit: Datasets Processing Streamlit App"

# Crear repositorio en GitHub y conectarlo
git remote add origin https://github.com/TU_USUARIO/datasets-processing.git

# Subir el código
git push -u origin main
```

### 2. Deploy en Streamlit Cloud

1. **Ir a [share.streamlit.io](https://share.streamlit.io)**

2. **Iniciar sesión** con tu cuenta de GitHub

3. **Click en "New app"**

4. **Configurar el deployment:**

   - Repository: `TU_USUARIO/datasets-processing`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL (opcional): `tu-app-personalizada` (si está disponible)

5. **Click en "Deploy!"**

6. **Esperar** (2-3 minutos) mientras Streamlit Cloud:
   - Clona tu repositorio
   - Instala las dependencias desde `requirements.txt`
   - Inicia la aplicación

### 3. URL de tu App

Tu aplicación estará disponible en:

```
https://TU_USUARIO-datasets-processing.streamlit.app
```

o

```
https://share.streamlit.io/TU_USUARIO/datasets-processing/main/app.py
```

## 📋 Checklist Pre-Deploy

- [x] `app.py` está en el directorio raíz
- [x] `requirements.txt` contiene todas las dependencias
- [x] Los archivos CSV están incluidos en el repositorio
- [x] `.gitignore` excluye archivos innecesarios
- [x] `README.md` documenta el proyecto

## 🔧 Configuración Avanzada (Opcional)

### Variables de Entorno

Si necesitas variables de entorno, créalas en Streamlit Cloud:

1. Ve a tu app en Streamlit Cloud
2. Click en "Settings" → "Secrets"
3. Agrega tus secretos en formato TOML

### Configuración de Tema

Crea `.streamlit/config.toml` en tu repositorio:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
headless = true
```

## 🐛 Troubleshooting

### Error: "Module not found"

- Verifica que todas las dependencias estén en `requirements.txt`
- Asegúrate de que las versiones sean compatibles

### Error: "File not found"

- Verifica que los archivos CSV estén en las carpetas correctas
- Asegúrate de que las rutas sean relativas, no absolutas

### App muy lenta

- Streamlit Cloud tiene recursos limitados en el plan gratuito
- Considera optimizar el código o usar caché (`@st.cache_data`)

## 💡 Tips para Optimización

### 1. Usar Cache

```python
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")
```

### 2. Lazy Loading

Cargar datos solo cuando se necesiten

### 3. Comprimir Imágenes

Reducir el tamaño de archivos estáticos

## 🔄 Actualizar la App

Para actualizar tu app después del deploy:

```bash
# Hacer cambios en tu código
git add .
git commit -m "Descripción de los cambios"
git push

# Streamlit Cloud actualizará automáticamente tu app
```

## 📊 Monitoreo

En Streamlit Cloud puedes:

- Ver logs de la aplicación
- Monitorear el uso de recursos
- Ver estadísticas de visitantes (con analytics)

## 🎯 URLs Útiles

- **Streamlit Cloud**: https://share.streamlit.io
- **Documentación**: https://docs.streamlit.io
- **Community Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Forum**: https://discuss.streamlit.io

---

## 📝 Ejemplo de Estructura Final

```
datasets_processing/
├── .streamlit/
│   └── config.toml
├── ejercicio_1/
│   ├── ejercicio1.py
│   └── Titanic-Dataset.csv
├── ejercicio_2/
│   ├── ejercicio2.py
│   └── student-mat.csv
├── ejercicio_3/
│   └── ejercicio3.py
├── .gitignore
├── app.py
├── requirements.txt
├── README.md
└── DEPLOYMENT.md (este archivo)
```

¡Listo para el deploy! 🚀
