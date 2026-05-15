from datetime import datetime
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from sklearn.cluster import KMeans
from scipy.stats import mode
from scipy.stats import ttest_1samp
from collections import Counter
import os
from PIL import Image
from PIL import  ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.stats import mode
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import pyttsx3
import os
import threading



def GaussianKernel(size_x,size_y,sigma):
    x,y=np.mgrid[-size_x//2 + 1:size_x//+1,-size_y//2+1:size_y//2 + 1]
    g=np.exp(-((x**2 + y**2)/(2.0*sigma**2)))

    return g/g.sum()

def Ilumination(img, mostrar=False):
    kernel_FG=GaussianKernel(9, 9, 2)
    imagen=np.pad(img, ((1, 1), (1, 1)), mode='median') 

    imagen_Gauss_FiltradaScipy=signal.convolve2d(imagen, kernel_FG, mode='same')
    imagen_Gauss_FiltradaScipy=imagen_Gauss_FiltradaScipy[1:-1, 1:-1]

    #Corrección de iluminación
    background =cv2.medianBlur(imagen_Gauss_FiltradaScipy.astype(np.uint8), 51)
    background=background.astype(np.float32)+1e-6 

    imagen_corregida=(imagen_Gauss_FiltradaScipy.astype(np.float32)/background)*255
    imagen_corregida=np.clip(imagen_corregida, 0, 255).astype(np.uint8)

    #Umbral de binarizado
    hist, bins = np.histogram(imagen_corregida, bins=256, range=(0, 256))
    total_pixeles=imagen_corregida.size
    cdf_directa=np.cumsum(hist)

    ##Toma el valor de gris tal que la suma acumulada de pixeles que van desde 0 hasta ese valor de gris, sea el 0.9% del total de pixeles.
    U=np.searchsorted(cdf_directa, total_pixeles*0.009) #

    #Binarizado
    binary_image=imagen_corregida.copy()
    for i in range(len(imagen_corregida)):
        for j in range(len(imagen_corregida[0])):
            if imagen_corregida[i][j]<U:
                binary_image[i][j]=255
            else:
                binary_image[i][j]=0

    return binary_image


def detectar_centroides(binarizada, imagen_org):
    if imagen_org.dtype != np.uint8:
        if imagen_org.max()<=1.0:
            imagen_org=(imagen_org*255).astype(np.uint8)
        else:
            imagen_org=imagen_org.astype(np.uint8)

    if len(imagen_org.shape)==2:
        imagen_color=cv2.cvtColor(imagen_org, cv2.COLOR_GRAY2BGR)
    else:
        imagen_color=imagen_org.copy()

    contornos, _ =cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroides = []
    for contorno in contornos:
        M=cv2.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            centroides.append((cx, cy))
            cv2.circle(imagen_color, (cx, cy), 5, (0, 0, 255), -1)
    return centroides


def erosion_dilatacion(img_bin_inv):
    kernel = np.ones((3, 3), np.uint8)
    kernel2=np.ones((2, 2), np.uint8)
    img_dilation = cv2.dilate(img_bin_inv, kernel, iterations=2)
    return img_dilation


def dilate(img_bin_inv, it=4):
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(img_bin_inv, kernel, iterations=it)
    return img_dilation

def maximos (ancho_alto):
  max_ancho=0
  max_alto=0
  for i in range(len(ancho_alto)):
    max_ancho=max(ancho_alto[i][0],max_ancho)
    max_alto=max(ancho_alto[i][1],max_alto)
  return max_ancho, max_alto

def contornos(imagen_binaria_inv, imagen):
    if imagen_binaria_inv.dtype!=np.uint8:
        imagen_binaria_inv=(imagen_binaria_inv>0).astype(np.uint8)*255

    imagen_pack = imagen.copy()
    cnts = cv2.findContours(imagen_binaria_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ancho_alto = []
    bounding_box_coord = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(imagen_pack, (x,y), (x+w, y+h), (36, 255, 12), 1)
        ancho_alto.append([w, h])
        ROI = imagen[y:y + h, x:x + w]
        bounding_box_coord.append([x, y, w, h])
    return cnts, ancho_alto, bounding_box_coord, imagen_pack


def contornos_maximos(imagen_binaria_inv, imagen,ancho_max, alto_max):
    if imagen_binaria_inv.dtype != np.uint8:
        imagen_binaria_inv=(imagen_binaria_inv>0).astype(np.uint8) * 255

    imagen_pack=imagen.copy()
    cnts=cv2.findContours(imagen_binaria_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if len(cnts)==2 else cnts[1]
    ancho_alto=[]
    bounding_box_coord=[]


    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(imagen_pack, (x, y), (x + ancho_max, y + alto_max), (36, 255, 12), 1)
        ancho_alto.append([ancho_max, alto_max])
        ROI = imagen[y:y + alto_max, x:x + ancho_max]
        bounding_box_coord.append([x, y, ancho_max,alto_max])

    return cnts, ancho_alto, bounding_box_coord, imagen_pack


def centroides_en_bounding_boxes(centroides, bounding_boxes):
    puntos_contenidos = {}
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        centroides_dentro = []

        for cx, cy in centroides:
            if x <= cx <= x+w and y <= cy <= y+h:
                centroides_dentro.append((cx, cy))

        puntos_contenidos[i] = {
            "centroides": centroides_dentro,
            "cantidad_centroides": len(centroides_dentro)
        }

    return puntos_contenidos

def bounding_boxes_centroides(bounding_boxes):
    centroides=[]
    for i, (x, y, w, h) in enumerate(bounding_boxes):
      centroides.append([(x+w/2),(y+h/2)])
    return centroides

def encontrar_boxes_con_centroides_compartidos(puntos_contenidos):
    boxes_with_shared_points = []

    for i, (box1, data1) in enumerate(puntos_contenidos.items()):
        for j, (box2, data2) in enumerate(puntos_contenidos.items()):
            if i<j:
                centroides1=data1["centroides"]
                centroides2=data2["centroides"]

                shared_points = set(centroides1).intersection(centroides2)
                if shared_points:

                    boxes_with_shared_points.append([
                        box1,
                        box2,
                        list(shared_points),
                        data1["cantidad_centroides"],
                        data2["cantidad_centroides"]
                    ])
    return boxes_with_shared_points

#De las cajas que comparten centroides, toma una caja 1, compara contra una caja 2, se cuenta la cantidad de centroides de cada una y
#elimina la que menos centroides tiene (porque la mayoria de las veces la caja incorrecta se superpone por uno o 2 puntos)

def eliminar_boxes_compartidos(boxes_with_shared_points, bounding_boxes):
    boxes_to_remove = set()
    for pair in boxes_with_shared_points:
        box1, box2, shared_points, cantidad1, cantidad2 = pair
        if cantidad1<cantidad2:
            boxes_to_remove.add(box1)
        elif cantidad2<cantidad1:
            boxes_to_remove.add(box2)

    updated_bounding_boxes = [bbox for i, bbox in enumerate(bounding_boxes) if i not in boxes_to_remove]

    return updated_bounding_boxes


def correcciones_alineamiento(boxes, tolerancia_y=2, ancho_imagen=None, alto_imagen=None):

    #Paso 1: cálculo del centroide de cada caja
    boxes_con_centroides=[]
    for (x,y,w,h) in boxes:
        cx=x+w//2
        cy=y+h//2
        boxes_con_centroides.append(((x,y,w,h), cx, cy))

    #Paso 2:Ordenado por coordenada y (de arriba a abajo)
    boxes_con_centroides.sort(key=lambda b: b[2])

    #Paso 3:Agrupación por filas
    filas=[]
    fila_actual=[]
    for box_info in boxes_con_centroides:
        if not fila_actual:
            fila_actual.append(box_info)
        else:
            _, _, cy_anterior=fila_actual[-1]
            _, _, cy_actual=box_info
            if abs(cy_actual-cy_anterior)<=tolerancia_y:
                fila_actual.append(box_info)
            else:
                filas.append(fila_actual)
                fila_actual=[box_info]
    if fila_actual:
        filas.append(fila_actual)
    #Paso 4:Corrección de alineamiento horizontal
    for i in range(len(filas)):
        if len(filas[i])<4:
            #Busqueda de fila anterior válida (debe tener 5 o más letras)
            for j in range(i-1,-1,-1):
                if len(filas[j])>= 5:
                    fila_valida=filas[j]
                    break
            else:
                fila_valida = None

            if fila_valida:
                for k in range(len(filas[i])):
                    x_actual, y_actual, w, h = filas[i][k][0]

                    #Componente de fila válida con X más cercana
                    comp_mas_cercana = min(
                        fila_valida, key=lambda comp: abs(comp[0][0]-x_actual)
                    )
                    altura_cercana = comp_mas_cercana[0][1]
                    nuevo_bbox = (x_actual, altura_cercana, w, h)
                    filas[i][k] = (nuevo_bbox, filas[i][k][1], filas[i][k][2])

    #Paso 5: Ordenamiento entre celdas
    for i in range(len(filas)):
        filas[i] = sorted(filas[i], key=lambda comp: comp[0][0])
    #Paso 6: lista con bounding boxes ordenados
    boxes_ordenados = []
    for fila in filas:
        for b in fila:
            boxes_ordenados.append(b[0])
    #Paso 7:eliminación de boxes inválidos
    boxes_filtrados = []
    for (x,y,w,h) in boxes_ordenados:
        if x==0 or y==0:
            continue
        if ancho_imagen is not None and x+w>ancho_imagen:
            continue
        if alto_imagen is not None and y+h>alto_imagen:
            continue
        boxes_filtrados.append((x,y,w,h))
    return boxes_filtrados


def insertar_espaciado(bounding_boxes,umbral_y=10):
    if not bounding_boxes:
        return []

    #Cálculo de moda
    distancias_x = []
    for i in range(1, len(bounding_boxes)):
        x_prev,y_prev,_, _=bounding_boxes[i-1]
        x_curr,y_curr,_, _=bounding_boxes[i]

        if abs(y_curr-y_prev)<umbral_y:
            distancias_x.append(abs(x_curr-x_prev))
    if not distancias_x:
        return bounding_boxes.copy()

    d_ref = mode(distancias_x, keepdims=False).mode

    #Espaciado entre celdas
    resultado = []
    for i in range(len(bounding_boxes) - 1):
        resultado.append(bounding_boxes[i])
        x_curr,y_curr,w_curr,h_curr=bounding_boxes[i]
        x_next,y_next,w_next,h_next=bounding_boxes[i + 1]
        d_x = abs(x_next-x_curr)

        if d_x>1.5*d_ref:
            resultado.append((bounding_boxes[i][0]+d_ref, y_curr, w_curr, h_curr))


    resultado.append(bounding_boxes[-1])
    resultado = list(set(resultado))
    resultado.sort(key=lambda b: (b[1], b[0]))
    return resultado


def eliminar_boxes_cercanos(boxes,w_max, umbral_y=10):
    boxes=sorted(boxes, key=lambda b:b[1])
    filas=[]
    umbral_x=w_max
    fila_actual=[boxes[0]]
    for b in boxes[1:]:
        if abs(b[1]-fila_actual[-1][1])<umbral_y:
            fila_actual.append(b)
        else:
            filas.append(fila_actual)
            fila_actual=[b]
    filas.append(fila_actual)
    resultado = []
    
    for fila in filas:
        fila = sorted(fila, key=lambda b: b[0])
        filtrados=[]
        for box in fila:
            x, y, w, h=box
            existe_cercano=False

            for b in filtrados:
                if abs(b[0]-x)<umbral_x:
                    existe_cercano=True
                    break
            if not existe_cercano:
                filtrados.append(box)
        resultado.extend(filtrados)
    return resultado

def Grayscale_a_RGB(img):
    if img.ndim == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img]*3, axis=-1)

    # Resize a 96x96
    img = cv2.resize(img, (96, 96))
    return img.astype('float32') / 255.0


def flatten_images(recortes):
    result = []
    for item in recortes:
        if isinstance(item, list):
            result.extend(flatten_images(item))
        else:
            result.append(item)
    return result

###Codigo principal Funcion
def Braille_dots(img, it, mostrar=False):
    #Corrección de iluminación
    imagen=np.pad(img,((1, 1), (1, 1)), mode='median')
    binary_image= Ilumination(img)
    binary_image = cv2.erode(binary_image, np.ones((2, 2), np.uint8), iterations=1)
    """binary_image= cv2.dilate(binary_image_ed, np.ones((3, 3), np.uint8), iterations=1)"""

    #Centroides
    centroides = detectar_centroides(binary_image, img)
    imagen_centroides = np.zeros(binary_image.shape)
    for i in range(len(centroides)):
        imagen_centroides[centroides[i][1], centroides[i][0]] =255


    #Dilatación de centroides

    centroides_dilate = dilate(imagen_centroides, it)

    #Bounding Boxes
    imagen_pack = img.copy()
    _, ancho_alto, bounding_box_coord, imagen_boxes = contornos(centroides_dilate, img)


    #Fijar parámetros máximos: h_max y w_max
    ancho_max, alto_max = maximos(ancho_alto)
   
    cnts, ancho_alto_maximos, bounding_box_coord_max, imagen_pack_max = contornos_maximos(
        centroides_dilate, img, ancho_max, alto_max
    )
    bounding_boxes_6pack = bounding_box_coord_max.copy()
    bounding_boxes_6pack.sort(key=lambda b: b[1])


    #Corrección de recortes superpuestos
    puntos_contenidos=centroides_en_bounding_boxes(centroides, bounding_boxes_6pack)

    boxes_with_shared_points=encontrar_boxes_con_centroides_compartidos(puntos_contenidos)

    bounding_boxes_actualizadas=eliminar_boxes_compartidos(boxes_with_shared_points, bounding_boxes_6pack)

    #Corrección de alineamiento horizontal
    bounding_boxes_actualizadas_b=correcciones_alineamiento(bounding_boxes_actualizadas, tolerancia_y=14, ancho_imagen=img.shape[1], alto_imagen=img.shape[0])

    #Obtención de centroides debounding boxes
    imagen_new_pack=imagen.copy()
    boxes=[]
    centroides_boxes=[]

    #Inserción de espaciados entre celdas
    act=insertar_espaciado(bounding_boxes_actualizadas_b, umbral_y=10)

    #vector final:Eliminación de celdas extras
    texto_braille=eliminar_boxes_cercanos(act, ancho_max, umbral_y=10)



    for (x_new, y_new, w_new, h_new) in texto_braille:
        cv2.rectangle(imagen_new_pack, (x_new, y_new), (x_new + w_new, y_new + h_new), (36, 255, 12), 1)
        ROI = imagen[y_new:y_new + h_new, x_new:x_new + w_new]
        boxes.append([ROI])

        # Cálculo de Centroides de Bounding Boxes
        cx=x_new+w_new//2
        cy=y_new+h_new// 2
        centroides_boxes.append((cx, cy))

    #Control
    plt.figure(figsize=(10, 10))
    plt.imshow(imagen_new_pack, cmap="gray")
    plt.axis('off')
    plt.show()


    return boxes, imagen_new_pack


def flatten_images(recortes):
    planos = []
    for grupo in recortes:
        for img in grupo:
            if img is not None:
                planos.append(img)
    return planos

def braille_num_convert(texto):
    mapa = {
        'a':'1','b':'2','c':'3','d':'4','e':'5',
        'f':'6','g':'7','h':'8','i':'9','j':'0'
    }

    resultado=""
    i=0

    while i<len(texto):
        char = texto[i]

        if char=='#' and i+1<len(texto) and texto[i + 1] in mapa:
            i += 1 

            while i<len(texto) and texto[i] in mapa:
                resultado += mapa[texto[i]]
                i+=1
        else:
            resultado += char
            i+=1

    return resultado


def algoritmo(img, clases, modelo, pdf_path,it, guardar=True):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img debe ser np.ndarray, no {type(img)}")


    #Redimensionamiento
    img_n=cv2.resize(img, (3000, 2100))
    boxes, imagen_new_pack = Braille_dots(img_n, it, mostrar=False)


    imagenes_planas = flatten_images(boxes)
    imagenes_procesadas = []
    for img in imagenes_planas:
        try:
            img_proc = Grayscale_a_RGB(img)
            imagenes_procesadas.append(img_proc)
        except Exception as e:
            print("Error procesando imagen:", e)
    X = np.array(imagenes_procesadas)
    print("recortes:",len(X))

    #Decodificación final
    idx_to_letra={}
    for idx, nombre in enumerate(clases):
        letra = nombre.split("_")[1]
        if letra== "":
            letra=" "
        idx_to_letra[idx] = letra

    predicciones=modelo.predict(X, verbose=0)
    clases_pred=np.argmax(predicciones, axis=1)

    texto="".join(idx_to_letra.get(i, "?") for i in clases_pred)

    #Correcciones numéricas
    texto_final=braille_num_convert(texto)
  
    #Generación de PDF
    os.makedirs(r"C:\Users\user\OneDrive\Desktop\PYTHON\Transcripcion", exist_ok=True)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    ancho, alto=A4
    margen=20*mm
    c.setFont("Helvetica", 12)
    y = alto - margen
    line_height = 14
    max_chars = 90
    for i in range(0, len(texto_final), max_chars):
        linea = texto_final[i:i+max_chars]
        c.drawString(margen, y, linea)
        y -= line_height
        if y < margen:
            c.showPage()
            c.setFont("Helvetica", 12)
            y=alto - margen
    c.save()
    return texto_final, imagen_new_pack



def Grayscale_a_RGB(img):
    if img.ndim == 2:
        img = np.stack((img,)*3, axis=-1)
    img = cv2.resize(img, (96, 96))
    return img.astype("float32") / 255.0


#-----------------------------------------------------------------------
##Modelo 

modelo = load_model(r"C:\Users\user\OneDrive\Desktop\PYTHON\modelo_braille_19_b.keras")

clases=['letra_ _000000','letra_#_001111','letra_+_011010','letra_=_011011','letra_-_001001','letra_._001000','letra_a_100000','letra_b_110000','letra_c_100100','letra_d_100110','letra_e_100010','letra_f_110100','letra_g_110110','letra_h_110010','letra_i_010100','letra_j_010110','letra_l_111000','letra_m_101100','letra_n_101110','letra_ñ_110111','letra_o_101010','letra_p_111100','letra_q_111110','letra_r_111010','letra_s_011100','letra_t_011110','letra_u_101001','letra_v_111001','letra_w_010111','letra_y_101111','letra_z_101011']

pdf_path = r"C:\Users\user\OneDrive\Desktop\Transcripcion_braille.pdf"

COLOR_BTN = "#FFD700"
FONT_BTN = ("Arial", 18, "bold")
FONT_TITLE = ("Arial", 26, "bold")
BG_COLOR = "black"
FG_COLOR = "white"

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def hablar(texto):
    engine.say(texto)
    engine.runAndWait()

def hablar_async(texto):
    threading.Thread(target=hablar, args=(texto,), daemon=True).start()

def limpiar_pantalla():
    for widget in root.winfo_children():
        widget.destroy()

def elegir_archivo():
    global ruta_imagen

    ruta = filedialog.askopenfilename(
        filetypes=[("Imagenes", "*.jpg *.png *.jpeg")]
    )

    if ruta:
        ruta_imagen = ruta
        procesar_imagen()

def procesar_imagen():
    global texto_generado
    try:
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        texto, debug_img = algoritmo(
            img,
            clases,
            modelo,
            pdf_temp,
            it=20,
            guardar=False
        )
        texto_generado = texto
        mostrar_slide2(texto)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def guardar_pdf():
    try:
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")

        ruta_guardado = filedialog.asksaveasfilename(
            initialdir=downloads_path,
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            title="Guardar PDF"
        )
        if not ruta_guardado:
            return
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        algoritmo(
            img,
            clases,
            modelo,
            ruta_guardado,
            it=20,
            guardar=False
        )
        messagebox.showinfo("Éxito", "PDF guardado correctamente")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def abrir_camara():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo acceder a la cámara")
        return

    hablar_async("Cámara activada")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(
            "Camara-ENTER/FOTO: capturar | ESC: salir",
            frame
        )
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 13 or key == 32:
            cv2.imwrite("foto_capturada.jpg", frame)
            messagebox.showinfo(
                "Foto guardada",
                "La imagen fue capturada correctamente"
            )
            break
    cap.release()
    cv2.destroyAllWindows()

def mostrar_slide1():
    limpiar_pantalla()
    frame = tk.Frame(root, bg=BG_COLOR)
    frame.pack(expand=True)
    titulo = tk.Label(
        frame,
        text="ASISTENTE BRAILLE",
        font=("Arial", 28, "bold"),
        bg=BG_COLOR,
        fg="#FFD700"
    )
    titulo.pack(pady=(20,10))

    subtitulo = tk.Label(
        frame,
        text="Seleccione una opción",
        font=("Arial", 16),
        bg=BG_COLOR,
        fg="white"
    )
    subtitulo.pack(pady=(0,30))

    btn_archivo = tk.Button(
        frame,
        text="Elegir archivo",
        bg=COLOR_BTN,
        fg="black",
        activebackground="#FFC300",
        font=("Arial", 16, "bold"),
        width=22,
        height=2,
        bd=0,
        relief="flat",
        cursor="hand2",
        command=elegir_archivo
    )
    btn_archivo.pack(pady=10)

    btn_archivo.bind("<Enter>", lambda e: hablar_async("Elegir archivo"))

    btn_camara = tk.Button(
        frame,
        text="Cámara",
        bg=COLOR_BTN,
        fg="black",
        font=FONT_BTN,
        width=19,
        height=2,
        bd=2,
        relief="solid",
        command=abrir_camara
    )
    btn_camara.pack(pady=10)

    btn_camara.bind("<Enter>", lambda e: hablar_async("Cámara"))

def mostrar_slide2(texto):
    limpiar_pantalla()
    titulo = tk.Label(
        root,
        bg=BG_COLOR,
        text="Texto detectado",
        fg="#FFD700",
        font=FONT_TITLE
    )
    titulo.pack(pady=20)
    text_box = tk.Text(root, height=10, width=35, font=("Arial", 14))
    text_box.pack(pady=10)
    text_box.tag_configure("center", justify='center')
    text_box.insert("1.0", texto, "center")

    btn_descargar = tk.Button(
        root,
        text="Descargar PDF",
        bg=COLOR_BTN,
        fg="black",
        font=FONT_BTN,
        width=20,
        height=2,
        bd=2,
        relief="solid",
        command=guardar_pdf
    )
    btn_descargar.pack(pady=20)
    btn_descargar.bind("<Enter>", lambda e: hablar_async("Descargar PDF"))

    def reproducir_texto():
        contenido = text_box.get("1.0", tk.END).strip()
        if contenido:
            hablar_async("Escuchar texto")
            hablar_async(contenido)

    btn_audio = tk.Button(
        root,
        text="Escuchar texto",
        bg=COLOR_BTN,
        fg="black",
        font=FONT_BTN,
        width=20,
        height=2,
        bd=2,
        relief="solid",
        command=reproducir_texto
    )
    btn_audio.pack(pady=10)

    btn_audio.bind("<Enter>", lambda e: hablar_async("Escuchar texto"))

root = tk.Tk()
root.title("Braille OCR")
root.geometry("400x600")
root.configure(bg=BG_COLOR)

ruta_imagen = None
texto_generado =""
pdf_temp="temp.pdf"

mostrar_slide1()

root.mainloop()