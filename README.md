# BrailleDecoderAlgorithm

Algoritmo para la decodificación automática de textos en Braille mediante técnicas de procesamiento de imágenes y redes neuronales convolucionales.

---

## Contenido del repositorio

1. `mainBraille.py`

   Archivo principal en Python que contiene la implementación completa del algoritmo decodificador de Braille.

   Funciones principales:

   - `algoritmo()`
     
     Función principal del sistema que integra segmentación, clasificación mediante red neuronal y generación del texto final, además de crear el archivo PDF.

   - `Braille_dots()`
     
     Función principal de segmentación que integra preprocesamiento, detección de centroides, generación y corrección de *bounding boxes* para obtener las celdas Braille.

   - `Ilumination()`
     
     Realiza la corrección de iluminación mediante filtrado gaussiano y normalización con el fondo, seguida de una binarización adaptativa para resaltar los puntos Braille.

   - `contornos()`
     
     Detecta contornos en la imagen y genera *bounding boxes* iniciales alrededor de cada componente conectado.

   - `contornos_maximos()`
     
     Ajusta las *bounding boxes* a un tamaño uniforme basado en las dimensiones máximas detectadas.

   - `encontrar_boxes_con_centroides_compartidos()`
     
     Identifica *bounding boxes* que comparten centroides, indicando posibles solapamientos.

   - `eliminar_boxes_compartidos()`
     
     Elimina *bounding boxes* redundantes en función de la cantidad de centroides contenidos.

   - `correcciones_alineamiento()`
     
     Organiza las *bounding boxes* en filas y columnas mediante corrección de alineamiento horizontal.

   - `insertar_espaciado()`
     
     Agrega espacios entre celdas en función de la distancia horizontal entre *bounding boxes*.

   - `braille_num_convert()`
     
     Convierte secuencias de caracteres Braille en su equivalente numérico según la notación estándar.

     Ejemplo:
     
     ```text
     #a = 1
     #b = 2
     #c = 3
     ```

---

2. `DATASETS`

   - `DATASET-COMPLETO`
     
     Dataset compuesto por 17.646 imágenes distribuidas en 35 clases.

   - `dataset-31-clases`
     
     Dataset final utilizado para el entrenamiento y validación de modelos, compuesto por 31 clases.

---

3. `Modelos_entrenados_1_5_summary.ipynb`

   Notebook de Google Colab con un resumen de resultados, métricas y desempeño de los modelos entrenados.
