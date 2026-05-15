# BrailleDecoderAlgorithm
Algoritmo Decodificador de Braille. 

Los archivos contenido son los siguientes:

1.   mainBraille.py: Archivo python con cuerpo de código del algoritmo decodificador de braille. Las principales funciones son:
   
  algoritmo(): función principal del sistema que integra segmentación, clasificación mediante red neuronal y generación del texto final, además de crear el archivo PDF.
  
  Braille_dots(): función principal de segmentación que integra preprocesamiento, detección de centroides, generación y corrección de bounding boxes para obtener las celdas Braille.
  Ilumination(): realiza la corrección de iluminación mediante filtrado gaussiano y normalización con el fondo, seguida de una binarización adaptativa para resaltar los puntos Braille.
  
  contornos(): detecta contornos en la imagen y genera bounding boxes iniciales alrededor de cada componente conectado.
  
  contornos_maximos(): ajusta las bounding boxes a un tamaño uniforme basado en las dimensiones máximas detectadas..
  
  encontrar_boxes_con_centroides_compartidos(): identifica bounding boxes que comparten centroides, lo que indica posibles solapamientos.
  
  eliminar_boxes_compartidos(): elimina bounding boxes redundantes en función de la cantidad de centroides contenidos.
  
  correcciones_alineamiento(): organiza las bounding boxes en filas y columnas. Corrección de alineamiento horizontal.
  
  insertar_espaciado(): agrega espacios entre celdas en función de la distancia horizontal entre bounding boxes.
  
  braille_num_convert(): convierte secuencias de caracteres Braille en su equivalente numérico según la notación estándar. (Ejp: #a = 1, #b = 2, # c = 3, etc.)

2. Datasets:
  
