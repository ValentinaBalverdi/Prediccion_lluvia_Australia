# Predicción de lluvia en Australia con Machine Learning   

Este repositorio contiene la resolución del Trabajo Práctico de Aprendizaje Automático I (TUIA–FCEIA).
El objetivo es predecir si lloverá al día siguiente en distintas ciudades de Australia utilizando el dataset weatherAUS.csv.

Se trabajó con scikit-learn, TensorFlow, PyCaret, SHAP y Docker para implementar un pipeline reproducible, comparar modelos de clasificación y desplegar la solución.

---

## ⚙️ Metodología  
1. **Preprocesamiento:** imputación de datos faltantes, codificación categórica, estandarización.  
2. **Clustering:** agrupación de ciudades en *regiones* a partir de latitud/longitud.  
3. **Modelos implementados:**  
   - Regresión logística.  
   - Redes neuronales (TensorFlow).  
   - AutoML con PyCaret.  
4. **Métricas evaluadas:** accuracy, precisión, recall, F1, matriz de confusión y curva ROC.  
5. **Explicabilidad:** análisis con SHAP (gráficas globales y locales).  
6. **MLOps:** serialización del pipeline, creación de contenedor Docker y script de inferencia.  

---


1. Clona el repositorio:
   ```bash
   git clone https://github.com/ValentinaBalverdi/Prediccion_lluvia_Australia.git
   ```

2. Genera el pipeline serializado (`pipeline.pkl`):
   ```bash
   python docker/create_pipeline.py
   ```
   > **Nota:** `pipeline.pkl` excede los 100 MB y por eso no se incluye en GitHub.

3. Desde la raíz del repositorio, construye la imagen:
    ```bash
    docker build   -t aa1-clasificacion:latest   -f docker/dockerfile .
    ```

4. Copia tu fichero de entrada (`input.csv`) en `docker/files/` (o usa el que ya está: `docker/files/input.csv`).

5. Ejecuta el contenedor, montando la carpeta `files` para las entradas y salidas:
    ```bash
    docker run --rm \
    -m 4g \
    -v "$(pwd)/docker/files:/app/docker/files" \
    -e MODEL_PKL=/app/docker/files/pipeline.pkl \
    -e INPUT_CSV=/app/docker/files/input.csv \
    -e OUTPUT_CSV=/app/docker/files/output.csv \
    aa1-clasificacion:latest
    ```
    - Esto leerá `docker/files/input.csv` dentro del contenedor.  
    - Las predicciones se volcarán en `docker/files/output.csv` en tu máquina.

6. Consulta los resultados en `docker/files/output.csv`.  

---
Autores:
- Valentina Balverdi
- Franco Caballero
- Valentin Rosito