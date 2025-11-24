import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_icon="🥈", page_title="Asistente Silver Economy")

if st.secrets.get("ESTADO_DEL_CHAT", "true") == "false":
    st.warning("🔒 Chat en mantenimiento.")
    st.stop()

st.title("🥈 Asistente Silver Economy (Híbrido)")

# --- GESTIÓN API KEY ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    key = st.sidebar.text_input("API Key:", type="password")
    if not key: st.stop()
    os.environ["OPENAI_API_KEY"] = key

client_audio = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- FUNCIONES DE AUDIO ---
def transcribir_audio(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"
        transcript = client_audio.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text
    except Exception as e:
        return None

def texto_a_voz(texto):
    try:
        response = client_audio.audio.speech.create(
            model="tts-1", voice="alloy", input=texto
        )
        return io.BytesIO(response.content)
    except Exception as e:
        return None

# --- CARGA DE DATOS ---
@st.cache_resource
def iniciar_base_datos():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_data = os.path.join(ruta_base, "Data")
    if not os.path.exists(ruta_data): return None
    
    docs = []
    with st.spinner("Cargando memoria..."):
        for archivo in os.listdir(ruta_data):
            if archivo.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(ruta_data, archivo))
                docs.extend(loader.load())
    
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
template_seguridad = """Analiza el mensaje y clasifica: 
1. PELIGRO (suicidio, autolesión, emergencias).
2. NEGATIVO (tristeza, soledad, enojo).
3. NORMAL (saludos, preguntas).
Responde SOLO con la palabra clave. Mensaje: {mensaje}"""
prompt_seguridad = ChatPromptTemplate.from_template(template_seguridad)

def analizar_riesgo(mensaje):
    return (prompt_seguridad | llm_seguridad).invoke({"mensaje": mensaje}).content.strip().upper()

llm_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
template_chat = """Eres un asistente virtual experto en Silver Economy, diseñado para acompañar a personas mayores y sus familias.
Tu prioridad es ser útil, pero sobre todo CÁLIDO, PACIENTE y RESPETUOSO.

Sigue estas reglas estrictas para responder:

1. 👋 SALUDOS (Prioridad Alta): Si el usuario te saluda (ej: "hola", "buenos días"), IGNORA el contexto de los documentos. Simplemente responde el saludo con amabilidad, preséntate y pregunta en qué puedes ayudar.
   * Ejemplo: "¡Hola! Es un gusto saludarte. Soy tu Asistente de Silver Economy. ¿Qué te gustaría saber hoy?"

2. ❤️ EMPATÍA Y TONO:
   * Usa frases conectoras amables: "Entiendo que esto es importante", "Gracias por tu pregunta", "Con mucho gusto te explico".
   * Usa un lenguaje sencillo y claro, evitando palabras demasiado técnicas.

3. 📄 USO DEL CONTEXTO:
   * Para responder preguntas de contenido, básate ÚNICAMENTE en la información del "Contexto" proporcionado abajo.
   * Si la respuesta está en el texto, explícala de forma conversacional, no como un robot leyendo una lista.

4. 🚫 SI NO LO SABES:
   * Si la información no está en el contexto, NO la inventes.
   * Discúlpate con elegancia: "Lamento decirte que no tengo información específica sobre ese punto en mis documentos actuales, pero estoy aquí para ayudarte con cualquier otro tema del archivo.".
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ultimo_audio_id" not in st.session_state:
    st.session_state.ultimo_audio_id = None

for msg in st.session_state.chat_history:
    st.chat_message(msg.type).write(msg.content)

# --- ZONA DE ENTRADA ---
col1, col2 = st.columns([1, 4])
with col1:
    st.write("🎤 Voz:")
    # El grabador nos da datos
    audio_data = mic_recorder(start_prompt="🔴", stop_prompt="⏹️", key='recorder')

# CAMBIO CLAVE: El input de texto SIEMPRE está visible (fuera de los IFs)
texto_input = st.chat_input("Escribe tu pregunta aquí...")

prompt_usuario = None
responder_con_voz = False

# LÓGICA DE PRIORIDAD:
# 1. ¿Hay un audio Y es diferente al último que procesamos? (Es NUEVO)
if audio_data and audio_data['id'] != st.session_state.ultimo_audio_id:
    texto_transcrito = transcribir_audio(audio_data['bytes'])
    if texto_transcrito:
        prompt_usuario = texto_transcrito
        responder_con_voz = True
        st.session_state.ultimo_audio_id = audio_data['id'] # ¡Marcamos como procesado!

# 2. Si no es audio nuevo, ¿hay texto?
elif texto_input:
    prompt_usuario = texto_input
    responder_con_voz = False

# --- PROCESAMIENTO ---
if prompt_usuario:
    st.session_state.chat_history.append(HumanMessage(content=prompt_usuario))
    
    if responder_con_voz: 
        st.chat_message("user").write(f"🗣️ {prompt_usuario}")
    else:
        st.chat_message("user").write(prompt_usuario)
    
    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            riesgo = analizar_riesgo(prompt_usuario)
            audio_out = None
            
            if "PELIGRO" in riesgo:
                respuesta = """🚨 **Mensaje Importante** 🚨
                
                Siento mucho que estés pasando por un momento tan difícil. No estás solo/a.
                Por favor, busca ayuda profesional inmediatamente.
                
                📞 **Línea de la Vida (Ejemplo):** 800-911-2000
                🏥 **Emergencias:** 112 / 911
                
                Aunque soy una IA y quiero ayudarte, en situaciones de crisis necesitas contacto humano urgente."""
                st.error("Emergencia detectada.")
            else:
                respuesta = responder_rag(prompt_usuario)
                if responder_con_voz:
                    audio_out = texto_a_voz(respuesta)
            
            st.write(respuesta)
            if audio_out:
                st.audio(audio_out, format="audio/mp3", autoplay=True)
            
    st.session_state.chat_history.append(AIMessage(content=respuesta))