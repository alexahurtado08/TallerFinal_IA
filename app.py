
# Importamos las librerías necesarias
import streamlit as st         # Librería para crear la interfaz web
import easyocr                 # Librería para el reconocimiento óptico de caracteres (OCR)
from PIL import Image          # Manejo de imágenes
from dotenv import load_dotenv # Carga de variables de entorno desde el archivo .env
import os                      # Acceso a variables del sistema
from groq import Groq          # Cliente para usar los modelos LLM de Groq
from huggingface_hub import InferenceClient  # Cliente para usar modelos LLM de Hugging Face
import numpy as np             # Manejo de arreglos numéricos
import cv2                     # Librería para procesamiento de imágenes (OpenCV)


#carga de variables de entorno

load_dotenv()  # Carga las claves del archivo .env
groq_api_key = os.getenv("GROQ_API_KEY")       # Clave de acceso para la API de Groq
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")  # Clave de acceso para la API de Hugging Face

# Si alguna de las claves falta, se muestra un error y la app se detiene
if not groq_api_key or not hf_api_key:
    st.error(" Faltan claves en tu archivo .env (GROQ_API_KEY o HUGGINGFACE_API_KEY).")
    st.stop()

# Se inicializan los clientes de cada proveedor con sus respectivas claves
cliente_groq = Groq(api_key=groq_api_key)
cliente_hf = InferenceClient(token=hf_api_key)


# configuracion de la apliccion

st.set_page_config(page_title="Taller IA: OCR + LLM", page_icon="🤖")
st.title(" Taller IA: OCR + LLM")
st.caption("Universidad EAFIT | Alumnos: Alexandra Hurtado y Mariana Valderrama | Proyecto: Aplicación Multimodal con OCR y LLMs")


# modulo 1

st.header("📸 Módulo 1: Lector de Imágenes (OCR)")
st.write("Sube una imagen con texto. Usamos *EasyOCR* para reconocer el texto contenido en ella.")

# Se usa @st.cache_resource 
@st.cache_resource
def cargar_lector():
    """Carga el modelo OCR una sola vez."""
    return easyocr.Reader(['es', 'en'])  # Modelos para español e inglés

lector = cargar_lector()  # Se carga el modelo en memoria (una sola vez)

# Se crea una variable de estado en Streamlit para guardar el texto extraído
if "texto_extraido" not in st.session_state:
    st.session_state.texto_extraido = ""

# El usuario puede subir una imagen en formato PNG, JPG o JPEG
archivo_imagen = st.file_uploader(" Sube una imagen (PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])

# Si se sube una imagen, se muestra y se procesa con EasyOCR
if archivo_imagen is not None:
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    # Convertimos la imagen a un arreglo de bytes para procesarla con OpenCV
    bytes_data = archivo_imagen.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Si la imagen se puede leer correctamente:
    if img_cv is not None:
        # Convertimos de BGR (OpenCV) a RGB (formato estándar)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # Se aplica OCR para extraer el texto de la imagen
        with st.spinner(" Extrayendo texto con OCR..."):
            resultados = lector.readtext(img_rgb)  # Devuelve texto detectado
            texto = "\n".join([res[1] for res in resultados])  # Se unen las líneas detectadas

        # Se guarda el texto extraído en la memoria de sesión
        st.session_state.texto_extraido = texto

        # Se muestra el texto en pantalla
        st.text_area(" Texto extraído:", texto, height=200)
    else:
        st.error(" No se pudo leer la imagen. Intenta con otra.")


# modulo 2

st.header(" Módulo 2: Análisis de Texto con LLMs")

# El usuario puede elegir el proveedor del modelo: GROQ o Hugging Face
proveedor = st.radio("Selecciona el proveedor:", ["GROQ", "Hugging Face"])

# Controles para ajustar los parámetros de generación del modelo
temperature = st.slider(" Creatividad (temperature)", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.slider(" Longitud máxima de respuesta (max_tokens)", 100, 1000, 300, 50)




# Configuración  groq

if proveedor == "GROQ":
    modelo = st.selectbox(
        "Selecciona el modelo de GROQ:",
        ["llama-3.1-8b-instant", "mixtral-8x7b"],
        index=0
    )

    tarea = st.selectbox(
        "Selecciona la tarea:",
        [
            "Solo extraer texto de la imagen",
            "Resumir en 3 puntos clave",
            "Traducir al inglés",
            "Identificar entidades principales",
            "Analizar el tono o sentimiento del texto"
        ]
    )

    if modelo != "llama-3.1-8b-instant":
        st.warning(" Este modelo puede no soportar todas las tareas(solo extraer texto). Usa *llama-3.1-8b-instant* para mejores resultados.")


# Configuración Hugging Face

else:
    modelo = st.selectbox(
        "Selecciona el modelo de Hugging Face:",
        ["mistralai/Mixtral-8x7B-Instruct-v0.1"],
        index=0
    )

    tarea = st.selectbox(
        "Selecciona la tarea:",
        [
            "Solo extraer texto de la imagen",
            "Resumir en 3 puntos clave",
            "Traducir al inglés",
            "Analizar sentimiento del texto"
        ]
    )


# boton para procesar texto 

if st.button(" Analizar Texto"):
    # Si no hay texto extraído, se muestra una advertencia
    if not st.session_state.texto_extraido:
        st.warning(" Primero sube una imagen y extrae el texto.")
    else:
        texto_entrada = st.session_state.texto_extraido  # Se recupera el texto del OCR

        with st.spinner(f" Procesando texto con {proveedor}..."):
            try:
                
                # OPCIÓN 1: GROQ 
                
                if proveedor == "GROQ":

                    # Si solo se quiere mostrar el texto original
                    if tarea == "Solo extraer texto de la imagen":
                        salida = texto_entrada

                    # Caso especial: Traducción con prompt optimizado
                    elif tarea == "Traducir al inglés":
                        # Prompt mejorado para obtener traducciones más precisas
                        prompt = (
                            "Eres un traductor profesional. Tu tarea es traducir el texto completamente al inglés. "
                            "No des explicaciones, solo devuelve la traducción final.\n\n"
                            f"Texto original:\n{texto_entrada}"
                        )

                        # Se llama al modelo de GROQ con el prompt
                        respuesta = cliente_groq.chat.completions.create(
                            model=modelo,
                            messages=[
                                {"role": "system", "content": "Eres un traductor profesional español-inglés."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        salida = respuesta.choices[0].message.content

                    # Otras tareas: resumen, análisis, etc.
                    else:
                        prompt = f"Debes {tarea.lower()}.\n\nTexto:\n{texto_entrada}"
                        respuesta = cliente_groq.chat.completions.create(
                            model=modelo,
                            messages=[
                                {"role": "system", "content": f"Eres un asistente experto en texto. Debes {tarea.lower()}."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        salida = respuesta.choices[0].message.content

                
                # OPCIÓN 2: HUGGING FACE 
                
                else:
                    if tarea == "Solo extraer texto de la imagen":
                        salida = texto_entrada
                    else:
                        # Se genera el prompt según la tarea seleccionada
                        if tarea == "Resumir en 3 puntos clave":
                            prompt = f"Resume el siguiente texto en 3 puntos clave en español:\n\n{texto_entrada}"
                        elif tarea == "Traducir al inglés":
                            prompt = f"Traduce el siguiente texto del español al inglés:\n\n{texto_entrada}"
                        elif tarea in ["Analizar sentimiento del texto", "Analizar el tono o sentimiento del texto"]:
                            prompt = f"Analiza el sentimiento del siguiente texto (positivo, negativo o neutral):\n\n{texto_entrada}"
                        else:
                            prompt = texto_entrada

                        # Se llama al modelo de Hugging Face en modo chat
                        respuesta = cliente_hf.chat.completions.create(
                            model=modelo,
                            messages=[
                                {"role": "system", "content": f"Eres un asistente que debe {tarea.lower()}."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )

                        salida = respuesta.choices[0].message["content"]

                # Se muestra la respuesta final del modelo en pantalla
                st.subheader(f" Resultado ({proveedor}):")
                st.markdown(str(salida))

            # Si ocurre un error en la conexión con las APIs
            except Exception as e:
                st.error(f" Error al conectar con {proveedor}: {e}")

# Mensaje final indicando que la app funciona correctamente
st.success(" Aplicación funcionando correctamente (OCR + LLMs).")
