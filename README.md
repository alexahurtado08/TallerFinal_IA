# TallerFinal_IA

Esta aplicación combina visión artificial y modelos de lenguaje (LLMs) para reconocer texto en imágenes y luego analizarlo usando inteligencia artificial.  

# Realizado por

**Autores:**  
- Alexandra Hurtado David  
- Mariana Valderrama Castañeda  

**Profesor:** Jorge Iván Padilla  
**Curso:** Inteligencia Artificial  
**Universidad:** EAFIT – 2025  

---

## Descripción general

El proyecto integra visión por computador con modelos de lenguaje para realizar análisis inteligentes de texto.  
Permite cargar una imagen, extraer su texto mediante OCR y procesarlo con modelos LLM para obtener diferentes tipos de resultados como resúmenes, traducciones o análisis de sentimiento.

---

## Funcionalidades principales

- Extraer texto desde una imagen mediante OCR (EasyOCR).  
- Analizar el texto usando modelos avanzados de lenguaje con GROQ o Hugging Face.  
- Realizar tareas como resumen, traducción, análisis de sentimientos o identificación de entidades.  
- Ajustar parámetros como la creatividad (temperature) o la longitud de la respuesta (max_tokens).  

---

## Tecnologías principales

- Python 3.10+  
- Streamlit (interfaz interactiva)  
- EasyOCR (reconocimiento de texto en imágenes)  
- OpenCV (procesamiento de imágenes)  
- Groq API (modelos de lenguaje rápidos y precisos)  
- Hugging Face API (modelos de lenguaje públicos y multitarea)  
- dotenv (manejo de claves y variables de entorno)  

---

## Instalación y configuración

1. Clona o descarga este repositorio.  

   ```bash
   git clone https://github.com/tu_usuario/TallerFinal_IA.git
   cd TallerFinal_IA
2. Crea un entorno virtual (Opcional pero recomendado)
   
    ```bash
    python -m venv env
    source env/bin/activate   #En Linux
    env\Scripts\activate      # En windows
    
3. Instala las dependecias necesarias

   ```bash
   pip install -r requirements.txt
   
4. Crea un archivo .env en la raíz del proyecto con tus claves de acceso.
   
   ```bash
   GROQ_API_KEY=tu_api_key_de_groq
   HUGGINGFACE_API_KEY=tu_api_key_de_hugging_face
   
5. Ejecuta la aplicación.
   ```bash
   streamlit run app.py

---
# Uso de la aplicación

1. Sube una imagen que contenga texto (por ejemplo, una hoja escaneada o una foto de un cartel).  
2. El sistema usa **EasyOCR** para leer y mostrar el texto detectado.  
3. Selecciona el proveedor del modelo:
   - **GROQ** (modelo `llama-3.1-8b-instant` recomendado por su velocidad y precisión).  
   - **Hugging Face** (modelo `mistralai/Mistral-7B-Instruct-v0.2`, estable y gratuito).  
4. Elige la tarea que deseas realizar:
   - Resumir en 3 puntos clave.  
   - Traducir al inglés.  
   - Analizar el tono o sentimiento del texto.  
   - Extraer texto directamente (sin procesarlo).  
5. Ajusta los parámetros de creatividad (`temperature`) y longitud (`max_tokens`).  
6. Presiona **"Analizar Texto"** y revisa el resultado generado en pantalla.  


---
# Puntos de discusión y reflexión final

## Diferencias de velocidad entre GROQ y Hugging Face
- GROQ fue mucho más rápido al dar las respuestas, mientras que Hugging Face tardó un poco más.  
- Esto ocurre porque GROQ tiene infraestructura optimizada para procesar modelos casi al instante, mientras que Hugging Face usa servidores compartidos.  
- En resumen: GROQ es más veloz y fluido, y Hugging Face más variable en tiempo de respuesta.  

## Cómo afecta el parámetro temperature
- El parámetro `temperature` define qué tan creativas o precisas son las respuestas obtenidas.  
- Valores bajos nos brindarán respuestas directas y constantes, ideal para análisis o traducción.  
- Valores altos no darán como resultados respuestas más creativas y variadas, útil para resúmenes o generación de ideas.  

## Importancia del texto extraído por el OCR
- La calidad del texto del OCR es muy importante, esto debido a que si tenemos un texto más limpio el modelo lo entiende bien y nos puede brindar respuestas más acertadas, 
Mientras que si obtenemos textos confusos o incompletos, el modelo también se confunde a la hora de brindar las respuestas a las tareas.
Por esto es muy importante la calidad del texto extraído y de la imagen procesada. 

## Qué más se podría integrar en la aplicación
Algunas cosas que se podrían integrar serían:
- Clasificación de textos según su tipo (noticia, correo, documento, etc.).  
- Posibilidad de realizar preguntas y obtener una respuesta. 
- Mejora de la redacción de textos OCR.  
- Traducciones a más idiomas.
- Identificación de lugares o nombres.
- Creación de imágenes a partir del texto.
- Expansión de la idea principal.
- Generación de textos tipo poesía a partir del texto extraído.
  





