import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
# --- IMPORTACIONES DE VOZ ---
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_icon="🥈", page_title="Asistente Silver Economy")

# --- 1. INTERRUPTOR DE MANTENIMIENTO ---
if st.secrets.get("ESTADO_DEL_CHAT", "true") == "false":
    st.warning("🔒 Chat en mantenimiento.")
    st.stop()

st.title("🥈 Asistente Silver Economy (Con Voz)")

# --- 2. GESTIÓN API KEY ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    key = st.sidebar.text_input("API Key:", type="password")
    if not key:
        st.info("Ingresa la API Key.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = key

# Cliente extra para funciones de audio (Whisper y TTS)
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
            model="tts-1",
            voice="alloy", # Opciones: alloy, echo, fable, onyx, nova, shimmer
            input=texto
        )
        return io.BytesIO(response.content)
    except Exception as e:
        return None

# --- 3. CARGA DE DATOS ---
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

if st.session_state.vectorstore is None:
    st.error("No se encontraron documentos en la carpeta Data.")
    st.stop()

st.session_state.retriever = st.session_state.vectorstore.as_retriever()

# --- 4. CEREBRO DE SENTIMIENTOS (EL PSICÓLOGO) 🧠 ---
llm_seguridad = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
template_seguridad = """Analiza el mensaje y clasifica: 
1. PELIGRO (suicidio, autolesión, emergencias).
2. NEGATIVO (tristeza, soledad, enojo).
3. NORMAL (saludos, preguntas).
Responde SOLO con la palabra clave. Mensaje: {mensaje}"""
prompt_seguridad = ChatPromptTemplate.from_template(template_seguridad)

def analizar_riesgo(mensaje):
    return (prompt_seguridad | llm_seguridad).invoke({"mensaje": mensaje}).content.strip().upper()

# --- 5. CEREBRO RESPONDEDOR (EL BIBLIOTECARIO) 📚 ---
llm_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
template_chat = """Eres un asistente Silver Economy CÁLIDO, PACIENTE y RESPETUOSO.
1. 👋 SALUDOS: Si saludan, responde amablemente sin usar contexto.
2. ❤️ EMPATÍA: Si hay quejas o tristeza, sé muy empático.
3. 📄 CONTEXTO: Responde basándote SOLO en el contexto. Si no sabes, dilo.

Contexto: {context}
Historial: {chat_history}
Pregunta: {question}
Respuesta:"""
prompt_chat = ChatPromptTemplate.from_template(template_chat)

def responder_rag(pregunta):
    docs = st.session_state.retriever.invoke(pregunta)
    contexto = "\n".join([d.page_content for d in docs])
    historial = "\n".join([f"{m.type}: {m.content}" for m in st.session_state.chat_history[-4:]])
    return (prompt_chat | llm_chat).invoke({"context": contexto, "chat_history": historial, "question": pregunta}).content

# --- 6. INTERFAZ DE CHAT MULTIMODAL (VOZ + TEXTO) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg.type).write(msg.content)

# ZONA DE ENTRADA
col1, col2 = st.columns([1, 4])
with col1:
    st.write("🎤 Hablar:")
    audio_grabado = mic_recorder(start_prompt="🔴", stop_prompt="⏹️", key='recorder')

prompt_usuario = None

# Prioridad: Si hay voz nueva, la usamos. Si no, miramos el texto.
if audio_grabado:
    texto_transcrito = transcribir_audio(audio_grabado['bytes'])
    if texto_transcrito: prompt_usuario = texto_transcrito
elif texto_input := st.chat_input("Escribe aquí..."):
    prompt_usuario = texto_input

if prompt_usuario:
    st.session_state.chat_history.append(HumanMessage(content=prompt_usuario))
    # Si fue por voz, mostramos lo que entendió la IA
    if audio_grabado: st.chat_message("user").write(prompt_usuario)
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando emociones..."):
            
            # A) ANÁLISIS DE RIESGO
            riesgo = analizar_riesgo(prompt_usuario)
            audio_para_reproducir = None
            
            # 🔴 CASO ROJO: RIESGO DE VIDA
            if "PELIGRO" in riesgo:
                respuesta = """🚨 **Mensaje Importante** 🚨
                Siento mucho que estés pasando por esto. Por favor, busca ayuda profesional inmediatamente.
                📞 **Línea de la Vida:** 800-911-2000 | 🏥 **Emergencias:** 112 / 911"""
                st.error("Protocolo de emergencia activado.")
                # NOTA: En caso de peligro, NO generamos audio para no ser invasivos.
            
            # 🟡 CASO AMARILLO: TRISTEZA
            elif "NEGATIVO" in riesgo:
                st.info("💡 Detecto que este tema es sensible. Te respondo con cuidado:")
                respuesta = responder_rag(prompt_usuario)
                audio_para_reproducir = texto_a_voz(respuesta)
                
            # 🟢 CASO VERDE: NORMAL
            else:
                respuesta = responder_rag(prompt_usuario)
                audio_para_reproducir = texto_a_voz(respuesta)
            
            st.write(respuesta)
            
            # Reproducir audio si corresponde
            if audio_para_reproducir:
                st.audio(audio_para_reproducir, format="audio/mp3", autoplay=True)
            
    st.session_state.chat_history.append(AIMessage(content=respuesta))