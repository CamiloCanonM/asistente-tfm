import streamlit as st
import os
import io
import pandas as pd
import base64
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

# --- CONFIGURACIÓN ---
st.set_page_config(page_icon="🥈", page_title="Asistente Conversacional KIVIA.AI")

if st.secrets.get("ESTADO_DEL_CHAT", "true") == "false":
    st.warning("🔒 Chat en mantenimiento.")
    st.stop()

st.title("🥈 Asistente Conversacional KIVIA.AI")

# --- GESTIÓN API KEY ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    key = st.sidebar.text_input("API Key:", type="password")
    if not key: st.stop()
    os.environ["OPENAI_API_KEY"] = key

client_audio = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- FUNCIONES MULTIMODALES ---
def transcribir_audio(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"
        return client_openai.audio.transcriptions.create(model="whisper-1", file=audio_file).text
    except: return None

def texto_a_voz(texto):
    try:
        response = client_openai.audio.speech.create(model="tts-1", voice="alloy", input=texto)
        return io.BytesIO(response.content)
    except: return None

def analizar_imagen(imagen_bytes):
    try:
        base64_image = base64.b64encode(imagen_bytes).decode('utf-8')
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un asistente visual para mayores. Describe medicamentos o lee textos con claridad."},
                {"role": "user", "content": [{"type": "text", "text": "¿Qué hay en la imagen?"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except: return "Error analizando imagen."

# --- 🧠 EL CEREBRO DE INGESTA DE DATOS (NUEVO) ---
@st.cache_resource
def iniciar_base_datos():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_data = os.path.join(ruta_base, "Data")
    if not os.path.exists(ruta_data): return None
    
    docs = []
    
    # 🔄 ROUTER DE ARCHIVOS (FILTRADO POR EXTENSIÓN)
    with st.spinner("Procesando PDFs, Excels y CSVs..."):
        for archivo in os.listdir(ruta_data):
            ruta_completa = os.path.join(ruta_data, archivo)
            
            try:
                # CASO 1: PDF
                if archivo.endswith(".pdf"):
                    loader = PyPDFLoader(ruta_completa)
                    docs.extend(loader.load())
                
                # CASO 2: CSV (Tablas separadas por comas)
                elif archivo.endswith(".csv"):
                    loader = CSVLoader(ruta_completa, encoding="utf-8")
                    docs.extend(loader.load())
                
                # CASO 3: EXCEL (.xlsx)
                elif archivo.endswith(".xlsx"):
                    # Usamos Unstructured para Excel (modo "elements" extrae mejor el texto)
                    loader = UnstructuredExcelLoader(ruta_completa, mode="elements")
                    docs.extend(loader.load())
                
                # CASO 4: TEXTO PLANO (.txt)
                elif archivo.endswith(".txt"):
                    loader = TextLoader(ruta_completa, encoding="utf-8")
                    docs.extend(loader.load())
                    
            except Exception as e:
                st.error(f"Error cargando {archivo}: {e}")
                continue # Si falla un archivo, sigue con el siguiente
    
    if not docs: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    return FAISS.from_documents(splits, OpenAIEmbeddings())

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = iniciar_base_datos()
if st.session_state.vectorstore is None: st.stop()
st.session_state.retriever = st.session_state.vectorstore.as_retriever()

# --- CEREBROS (Psicólogo y Bibliotecario) ---
llm_seguridad = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
template_seguridad = """Actúa como un sistema de seguridad y clasificación de intenciones.
Analiza el siguiente mensaje y clasifícalo en una de estas 3 categorías estrictas:

1. PELIGRO: ÚNICAMENTE si hay intenciones claras de suicidio, autolesión, sobredosis intencional o violencia extrema.
2. NEGATIVO: Si el usuario expresa tristeza, soledad, depresión o malestar emocional, pero SIN riesgo de vida inminente.
3. NORMAL: Cualquier pregunta sobre salud, horarios de medicamentos, dosis, gestión financiera, saludos, o consultas de información general.

Mensaje del usuario: {mensaje}

Clasificación (Responde solo con una palabra):"""
prompt_seguridad = ChatPromptTemplate.from_template(template_seguridad)

def analizar_riesgo(mensaje):
    return (prompt_seguridad | llm_seguridad).invoke({"mensaje": mensaje}).content.strip().upper()

llm_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
template_chat = """Eres un asistente virtual experto en Silver Economy, diseñado para acompañar a personas mayores y sus familias.
Tu prioridad es ser útil, pero sobre todo CÁLIDO, PACIENTE y RESPETUOSO.

Sigue estas reglas estrictas para responder:

1. 👋 SALUDOS (Prioridad Alta): Si el usuario te saluda (ej: "hola", "buenos días"), IGNORA el contexto de los documentos. Simplemente responde el saludo con amabilidad, preséntate y pregunta en qué puedes ayudar.
   * Ejemplo: "¡Hola! Es un gusto saludarte. Soy tu Asistente Conversacional KIVIA.AI. ¿Qué te gustaría saber hoy?"

2. ❤️ EMPATÍA Y TONO:
   * Usa frases conectoras amables: "Entiendo que esto es importante", "Gracias por tu pregunta", "Con mucho gusto te explico".
   * Usa un lenguaje sencillo y claro, evitando palabras demasiado técnicas.

3. 📄 USO DEL CONTEXTO:
   * Para responder preguntas de contenido, básate ÚNICAMENTE en la información del "Contexto" proporcionado abajo.
   * Si la respuesta está en el texto, explícala de forma conversacional, no como un robot leyendo una lista.

4. 🚫 SI NO LO SABES:
   * Si la información no está en el contexto, NO la inventes.
   * Discúlpate con elegancia: "Lamento decirte que no tengo información específica sobre ese punto en mis documentos actuales, pero estoy aquí para ayudarte con cualquier otro tema del archivo."
5. responde en el idioma que el usuario pregunte.
Contexto: {context}
Historial: {chat_history}
Pregunta: {question}
Respuesta Amable:"""
prompt_chat = ChatPromptTemplate.from_template(template_chat)

def responder_rag(pregunta):
    docs = st.session_state.retriever.invoke(pregunta)
    contexto = "\n".join([d.page_content for d in docs])
    historial = "\n".join([f"{m.type}: {m.content}" for m in st.session_state.chat_history[-4:]])
    return (prompt_chat | llm_chat).invoke({"context": contexto, "chat_history": historial, "question": pregunta}).content

# --- INTERFAZ ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "ultimo_audio_id" not in st.session_state: st.session_state.ultimo_audio_id = None

# ZONA DE CÁMARA
with st.expander("📸 Cámara (Lectura de Etiquetas)"):
    imagen_capturada = st.camera_input("Foto")

# MOSTRAR CHAT
for msg in st.session_state.chat_history:
    st.chat_message(msg.type).write(msg.content)

# ENTRADA DE DATOS
col1, col2 = st.columns([1, 4])
with col1:
    st.write("🎤")
    audio_data = mic_recorder(start_prompt="🔴", stop_prompt="⏹️", key='recorder')
texto_input = st.chat_input("Escribe aquí...")

# LÓGICA DE PROCESAMIENTO
prompt_usuario = None
respuesta_ia = None
es_vision = False
responder_con_voz = False

# 1. Visión (Prioridad 1)
if imagen_capturada:
    if "ultima_foto_proc" not in st.session_state or st.session_state.ultima_foto_proc != imagen_capturada.getvalue():
        prompt_usuario = "📸 [Analizando imagen...]"
        with st.spinner("👁️ KIVIA está mirando..."):
            respuesta_ia = analizar_imagen(imagen_capturada.getvalue())
        es_vision = True
        st.session_state.ultima_foto_proc = imagen_capturada.getvalue()

# 2. Audio (Prioridad 2)
elif audio_data and audio_data['id'] != st.session_state.ultimo_audio_id:
    texto = transcribir_audio(audio_data['bytes'])
    if texto:
        prompt_usuario = texto
        responder_con_voz = True
        st.session_state.ultimo_audio_id = audio_data['id']

# 3. Texto (Prioridad 3)
elif texto_input:
    prompt_usuario = texto_input

# --- 🚦 APLICACIÓN DEL SEMÁFORO 🚦 ---
if prompt_usuario:
    st.session_state.chat_history.append(HumanMessage(content=prompt_usuario))
    if not es_vision: st.chat_message("user").write(f"🗣️ {prompt_usuario}" if responder_con_voz else prompt_usuario)

    # Si NO es visión, pasamos por el filtro de seguridad
    if not es_vision and not respuesta_ia:
        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                
                # 1. ANÁLISIS DE RIESGO
                riesgo = analizar_riesgo(prompt_usuario)
                
                # 🔴 CASO ROJO: PELIGRO
                if "PELIGRO" in riesgo:
                    respuesta_ia = """🚨 **ALERTA DE SEGURIDAD** 🚨
                    
                    He detectado una situación de riesgo vital.
                    KIVIA no puede atender emergencias críticas.
                    
                    📞 **Por favor, llama YA al 112 o al teléfono de la esperanza.**
                    No estás solo/a."""
                    st.error("Protocolo de suicidio/riesgo activado.")
                    responder_con_voz = False # No hablar para no agobiar
                
                # 🟡 CASO AMARILLO: TRISTEZA (Empatía Extra)
                elif "NEGATIVO" in riesgo:
                    st.info("💛 KIVIA detecta que te sientes mal. Activando modo Acompañamiento.")
                    # Agregamos una nota al prompt para que sea más cariñoso
                    prompt_usuario = f"[USUARIO TRISTE] {prompt_usuario}" 
                    respuesta_ia = responder_rag(prompt_usuario)
                
                # 🟢 CASO VERDE: NORMAL
                else:
                    respuesta_ia = responder_rag(prompt_usuario)

    # MOSTRAR Y GUARDAR RESPUESTA
    if respuesta_ia:
        if not es_vision: # Si fue visión ya se mostró arriba
            with st.chat_message("assistant"):
                st.write(respuesta_ia)
                if responder_con_voz:
                    audio_out = texto_a_voz(respuesta_ia)
                    if audio_out: st.audio(audio_out, format="audio/mp3", autoplay=True)

        st.session_state.chat_history.append(AIMessage(content=respuesta_ia))
        if es_vision: st.rerun()