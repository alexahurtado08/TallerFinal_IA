# =========================================
# TALLER FINAL - APLICACIÓN MULTIMODAL (OCR + LLMs)
# Versión FINAL 2025 - Mejor traducción con GROQ (Prompt optimizado)
# =========================================

import streamlit as st
import easyocr
from PIL import Image
from dotenv import load_dotenv
import os
from groq import Groq
from huggingface_hub import InferenceClient
import numpy as np
import cv2

# -----------------------------------------
# CARGA DE VARIABLES DE ENTORNO
# -----------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

if not groq_api_key or not hf_api_key:
    st.error("❌ Faltan claves en tu archivo .env (GROQ_API_KEY o HUGGINGFACE_API_KEY).")
    st.stop()

# Inicializar clientes
cliente_groq = Groq(api_key=groq_api_key)
cliente_hf = InferenceClient(token=hf_api_key)

# -----------------------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# -----------------------------------------
st.set_page_config(page_title="Taller IA: OCR + LLM", page_icon="🤖")
st.title("🧠 Taller IA: OCR + LLM")
st.caption("Universidad EAFIT | Alumnos: Alexandra Hurtado y Mariana Valderrama | Proyecto: Aplicación Multimodal con OCR y LLMs")

# -----------------------------------------
# MÓDULO 1: OCR
# -----------------------------------------
st.header("📸 Módulo 1: Lector de Imágenes (OCR)")
st.write("Sube una imagen con texto. Usamos *EasyOCR* para reconocer el texto contenido en ella.")

@st.cache_resource
def cargar_lector():
    """Carga el modelo OCR una sola vez."""
    return easyocr.Reader(['es', 'en'])

lector = cargar_lector()

if "texto_extraido" not in st.session_state:
    st.session_state.texto_extraido = ""

archivo_imagen = st.file_uploader("📂 Sube una imagen (PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])

if archivo_imagen is not None:
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    bytes_data = archivo_imagen.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_cv is not None:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        with st.spinner("🔍 Extrayendo texto con OCR..."):
            resultados = lector.readtext(img_rgb)
            texto = "\n".join([res[1] for res in resultados])
        st.session_state.texto_extraido = texto
        st.text_area("📄 Texto extraído:", texto, height=200)
    else:
        st.error("❌ No se pudo leer la imagen. Intenta con otra.")

# -----------------------------------------
# MÓDULO 2: ANÁLISIS CON LLMs
# -----------------------------------------
st.header("🤖 Módulo 2: Análisis de Texto con LLMs")

proveedor = st.radio("Selecciona el proveedor:", ["GROQ", "Hugging Face"])

temperature = st.slider("🌡 Creatividad (temperature)", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.slider("📝 Longitud máxima de respuesta (max_tokens)", 100, 1000, 300, 50)

# GROQ
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
        st.warning("⚠ Este modelo puede no soportar todas las tareas. Usa *llama-3.1-8b-instant* para mejores resultados.")

# HUGGING FACE
else:
    modelo = st.selectbox(
        "Selecciona el modelo de Hugging Face:",
        [
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
        ],
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

# -----------------------------------------
# BOTÓN PARA PROCESAR
# -----------------------------------------
if st.button("🚀 Analizar Texto"):
    if not st.session_state.texto_extraido:
        st.warning("⚠ Primero sube una imagen y extrae el texto.")
    else:
        texto_entrada = st.session_state.texto_extraido

        with st.spinner(f"🧠 Procesando texto con {proveedor}..."):
            try:
                # ----------------------------
                # OPCIÓN 1: GROQ (con mejor prompt de traducción)
                # ----------------------------
                if proveedor == "GROQ":
                    if tarea == "Solo extraer texto de la imagen":
                        salida = texto_entrada

                    elif tarea == "Traducir al inglés":
                        # ✅ Prompt optimizado para traducción
                        prompt = (
                            "Eres un traductor profesional. Tu tarea es traducir el texto completamente al inglés. "
                            "No des explicaciones, solo devuelve la traducción final.\n\n"
                            f"Texto original:\n{texto_entrada}"
                        )
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

                    else:
                        # Otras tareas con prompts normales
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

                # ----------------------------
                # OPCIÓN 2: HUGGING FACE (100% funcional conversacional)
                # ----------------------------
                else:
                    if tarea == "Solo extraer texto de la imagen":
                        salida = texto_entrada
                    else:
                        if tarea == "Resumir en 3 puntos clave":
                            prompt = f"Resume el siguiente texto en 3 puntos clave en español:\n\n{texto_entrada}"
                        elif tarea == "Traducir al inglés":
                            prompt = f"Traduce el siguiente texto del español al inglés:\n\n{texto_entrada}"
                        elif tarea in ["Analizar sentimiento del texto", "Analizar el tono o sentimiento del texto"]:
                            prompt = f"Analiza el sentimiento del siguiente texto (positivo, negativo o neutral):\n\n{texto_entrada}"
                        else:
                            prompt = texto_entrada

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

                # Mostrar resultado final
                st.subheader(f"🧩 Resultado ({proveedor}):")
                st.markdown(str(salida))

            except Exception as e:
                st.error(f"❌ Error al conectar con {proveedor}: {e}")

st.success("✅ Aplicación funcionando correctamente (OCR + LLMs).")
