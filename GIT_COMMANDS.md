# Comandos Git para Deployment

## Inicializar el Repositorio

```bash
# Inicializar Git
git init

# Agregar todos los archivos
git add .

# Primer commit
git commit -m "Initial commit: Datasets Processing Streamlit App"
```

## Conectar con GitHub

```bash
# Crear el repositorio en GitHub primero en: https://github.com/new
# Luego conectarlo:

git remote add origin https://github.com/TU_USUARIO/datasets-processing.git

# Verificar la conexión
git remote -v

# Subir el código
git push -u origin main
```

## Comandos Útiles

```bash
# Ver el estado de los archivos
git status

# Ver el historial de commits
git log --oneline

# Crear una nueva rama
git checkout -b feature/nueva-caracteristica

# Cambiar entre ramas
git checkout main

# Ver las diferencias
git diff

# Deshacer cambios no guardados
git checkout -- archivo.py

# Actualizar desde GitHub
git pull origin main
```

## Workflow Típico

```bash
# 1. Hacer cambios en el código
# 2. Ver qué cambió
git status

# 3. Agregar cambios
git add .
# o agregar archivos específicos
git add app.py requirements.txt

# 4. Hacer commit
git commit -m "Descripción clara de los cambios"

# 5. Subir a GitHub
git push

# 6. Streamlit Cloud actualizará automáticamente tu app
```

## Ignorar Archivos

El archivo `.gitignore` ya está configurado para ignorar:

- Entornos virtuales (venv/)
- Archivos de caché (**pycache**/)
- Configuraciones del IDE (.vscode/, .idea/)
- Archivos del sistema (.DS_Store, Thumbs.db)

## Solucionar Problemas Comunes

### Error: "fatal: not a git repository"

```bash
git init
```

### Error: "Updates were rejected"

```bash
# Forzar push (cuidado, sobrescribe el remoto)
git push -f origin main

# O mejor, hacer pull primero
git pull origin main --allow-unrelated-histories
git push origin main
```

### Cambiar el mensaje del último commit

```bash
git commit --amend -m "Nuevo mensaje"
```

### Deshacer el último commit (mantener cambios)

```bash
git reset --soft HEAD~1
```

### Ver archivos que serán ignorados

```bash
git status --ignored
```

## Crear un .gitignore personalizado

Si necesitas ignorar archivos adicionales, edita `.gitignore`:

```bash
# Archivo de ejemplo
mi_archivo_secreto.txt
*.log
datos_privados/
```

## Tags para Versiones

```bash
# Crear un tag
git tag -a v1.0 -m "Version 1.0 - Primera versión estable"

# Ver tags
git tag

# Subir tags a GitHub
git push origin --tags
```

## Recursos Útiles

- **GitHub Desktop**: https://desktop.github.com/ (GUI para Git)
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf
- **Learn Git Branching**: https://learngitbranching.js.org/
