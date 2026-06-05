# Proyecto Final: Minería de Datos Aplicada a la Salud Pública de México

* **Documentación Interactiva (GitHub Pages):** [https://danielasanluis.github.io/ProyectoFinalAMdD/]

NOTA: El dataset no se subio, pero el link se encuentra en el reporte (pdf).

## Instrucciones de Instalación y Ejecución

Este proyecto está desarrollado modularmente en Python utilizando un entorno de programación orientada a objetos (POO) y documentado de forma interactiva con Quarto. Sigue estos pasos en tu terminal de Linux para replicar el entorno localmente.

### Prerrequisitos
Asegúrate de tener instalado Python 3.10 o superior y la herramienta Quarto en tu sistema CLI.

### 1. Instalación y Configuración del Entorno

Primero, clona este repositorio en tu máquina local y navega a la carpeta raíz del proyecto:

- git clone [https://github.com/danielasanluis/ProyectoFinalAMdD.git]
- cd ProyectoFinalAMdD

Crea un entorno virtual aislado para evitar conflictos de librerías y actívalo:
- python3 -m venv venv
- source venv/bin/activate

Instala todas las dependencias y librerías necesarias del proyecto utilizando el archivo de requerimientos:
- pip install --upgrade pip
- pip install -r requirements.txt

Ejecucion de la carpeta src/ dentro de la misma.
1. python3 EdaAnalyzer.py
2. python3 ModelTrainer.py
3. python3 Clustering.py

---

### 2. Ejecución de los Componentes

#### Opción A: Ejecución de las celdas de código (Notebooks)
Para interactuar con el flujo de análisis o volver a entrenar los modelos, abre tu entorno de Jupyter o VS Code y ejecuta las libretas de forma secuencial desde la carpeta `notebooks/`:
1. `notebooks/EDA.ipynb` (Fase exploratoria y cálculo de IQR)
2. `notebooks/Modelo.ipynb` (Entrenamiento y persistencia del Random Forest)
3. `notebooks/Clustering.ipynb` (Segmentación K-Means y perfiles clínicos)

#### Opción B: Despliegue de la Documentación Interactiva Local (Quarto)
Si deseas renderizar localmente todo el portal interactivo y previsualizar las páginas HTML que se crearon en la carpeta `docs/`, ejecuta el servidor de Quarto con el siguiente comando:
```bash
quarto preview
```
El portal web interactivo se abrirá automáticamente en tu navegador predeterminado bajo la dirección local `http://localhost:6022/`.