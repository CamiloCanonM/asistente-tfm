import streamlit as st
import os
import io
import pandas as pd
import base64
import time
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

# --- CONFIGURACIÓN ---
st.set_page_config(page_icon="🥈", page_title="ASISTENTE KIVIA.AI")

if st.secrets.get("ESTADO_DEL_CHAT", "true") == "false":
    st.warning("🔒 Chat en mantenimiento.")
    st.stop()

st.title("🥈 KIVIA.AI (Ecosistema Completo)")

# --- GESTIÓN API KEY ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    key = st.sidebar.text_input("API Key:", type="password")
    if not key: st.stop()
    os.environ["OPENAI_API_KEY"] = key

client_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- FUNCIONES AUXILIARES ---
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
            model="gpt-4o-mini", 
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Describe esta imagen (medicamento/instrucciones)."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e: return f"Error: {e}"

def leer_reloj_en_vivo():
    # 👇👇👇 ¡PEGA AQUÍ TU LINK DE GOOGLE SHEETS (CSV)! 👇👇👇
    url_sheet = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5BW0ZT3Mp5Sd9DdpmAKqgPC-iZzrGyRIM7zV-_gcBTw8eR3SJAqklacU462M5QtB8qhVUG7Q38Hw_/pub?output=csv"
    
    try:
        if "PON_AQUI" in url_sheet: return None # Protección si no has puesto el link
        df = pd.read_csv(url_sheet)
        return df.iloc[-1] # Devuelve la última fila
    except: return None

# --- CARGA DE DATOS (RAG) ---
@st.cache_resource
def iniciar_base_datos():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_data = os.path.join(ruta_base, "Data")
    if not os.path.exists(ruta_data): return None
    docs = []
    with st.spinner("Cargando memoria..."):
        for archivo in os.listdir(ruta_data):
            ruta_path = os.path.join(ruta_data, archivo)
            try:
                if archivo.endswith(".pdf"): docs.extend(PyPDFLoader(ruta_path).load())
                elif archivo.endswith(".xlsx"):
                    df = pd.read_excel(ruta_path)
                    docs.append(Document(page_content=df.to_string(index=False), metadata={"source": archivo}))
            except: pass
    if not docs: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return FAISS.from_documents(splits, OpenAIEmbeddings())

from langchain.docstore.document import Document # Import necesario
if "vectorstore" not in st.session_state: st.session_state.vectorstore = iniciar_base_datos()
if st.session_state.vectorstore is None: st.stop()
st.session_state.retriever = st.session_state.vectorstore.as_retriever()

# --- CEREBROS (Seguridad y Chat) ---
llm_seguridad = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt_seguridad = ChatPromptTemplate.from_template("Clasifica: 1. PELIGRO, 2. NEGATIVO, 3. NORMAL. Mensaje: {mensaje}")
def analizar_riesgo(mensaje):
    return (prompt_seguridad | llm_seguridad).invoke({"mensaje": mensaje}).content.strip().upper()

llm_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
prompt_chat = ChatPromptTemplate.from_template("Eres KIVIA. Responde con calidez.\nContexto: {context}\nPregunta: {question}")
def responder_rag(pregunta):
    docs = st.session_state.retriever.invoke(pregunta)
    contexto = "\n".join([d.page_content for d in docs])
    return (prompt_chat | llm_chat).invoke({"context": contexto, "question": pregunta}).content

# --- INTERFAZ ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "ultimo_audio_id" not in st.session_state: st.session_state.ultimo_audio_id = None

# ⌚ BARRA LATERAL IOT (MANUAL)
with st.sidebar:
    st.header("⌚ Monitor Wearable")
    if st.button("🔄 Sincronizar Reloj"):
        datos = leer_reloj_en_vivo()
        if datos is not None:
            ritmo = int(datos.get('Ritmo', 70))
            pasos = int(datos.get('Pasos', 0))
            st.metric("❤️ Ritmo", f"{ritmo} bpm", delta=f"{ritmo-70}")
            st.metric("👣 Pasos", f"{pasos}")
            
            if ritmo > 100:
                st.session_state.iot_alert = f"ALERTA CRÍTICA: Ritmo {ritmo} bpm detectado por el reloj."
                st.error("⚠️ ANOMALÍA DETECTADA")
            else:
                st.success("✅ Signos estables")
                if "iot_alert" in st.session_state: del st.session_state.iot_alert
        else:
            st.warning("Configura el Link de Google Sheet en el código.")

# 📸 CÁMARA
with st.expander("📸 Cámara (Lectura de Etiquetas)"):
    imagen_capturada = st.camera_input("Foto")

# 💬 CHAT
for msg in st.session_state.chat_history:
    st.chat_message(msg.type).write(msg.content)

# 🎤 ENTRADA
col1, col2 = st.columns([1, 4])
with col1:
    st.write("🎤")
    audio_data = mic_recorder(start_prompt="🔴", stop_prompt="⏹️", key='recorder')
texto_input = st.chat_input("Escribe aquí...")

# 🧠 LÓGICA PROCESAMIENTO
prompt_usuario = None
respuesta_ia = None
es_vision = False
responder_con_voz = False
es_evento_iot = False

# 0. ALERTA IOT (Prioridad Máxima)
if "iot_alert" in st.session_state and st.session_state.iot_alert:
    prompt_usuario = st.session_state.iot_alert
    es_evento_iot = True
    del st.session_state.iot_alert 

# 1. VISIÓN (Si hay foto nueva)
elif imagen_capturada:
    if "ultima_foto_proc" not in st.session_state: st.session_state.ultima_foto_proc = None
    if imagen_capturada.getvalue() != st.session_state.ultima_foto_proc:
        prompt_usuario = "📸 [Analizando imagen...]"
        with st.spinner("👁️ KIVIA está mirando..."):
            respuesta_ia = analizar_imagen(imagen_capturada.getvalue())
        es_vision = True
        st.session_state.ultima_foto_proc = imagen_capturada.getvalue()

# 2. AUDIO (Si hay audio nuevo)
elif audio_data and audio_data['id'] != st.session_state.ultimo_audio_id:
    texto = transcribir_audio(audio_data['bytes'])
    if texto:
        prompt_usuario = texto
        responder_con_voz = True
        st.session_state.ultimo_audio_id = audio_data['id']

# 3. TEXTO
elif texto_input:
    prompt_usuario = texto_input

# EJECUCIÓN
if prompt_usuario:
    if es_evento_iot:
        st.chat_message("user", avatar="⌚").write(f"**WEARABLE:** {prompt_usuario}")
    else:
        st.session_state.chat_history.append(HumanMessage(content=prompt_usuario))
        if not es_vision: st.chat_message("user").write(f"🗣️ {prompt_usuario}" if responder_con_voz else prompt_usuario)

    if not es_vision and not respuesta_ia:
        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                if es_evento_iot and "ALERTA" in prompt_usuario:
                     respuesta_ia = "🚨 **ALERTA MÉDICA** 🚨\n\nEl reloj ha detectado una anomalía cardíaca severa.\n1. Siéntate.\n2. Contactando emergencias (112)."
                     st.error("PROTOCOLO DE EMERGENCIA")
                     responder_con_voz = True
                else:
                    riesgo = analizar_riesgo(prompt_usuario)
                    if "PELIGRO" in riesgo:
                        respuesta_ia = "🚨 EMERGENCIA: Llama al 112. No estás solo."
                        st.error("Alerta de seguridad.")
                        responder_con_voz = False
                    elif "NEGATIVO" in riesgo:
                        respuesta_ia = responder_rag(f"[USUARIO TRISTE] {prompt_usuario}")
                    else:
                        respuesta_ia = responder_rag(prompt_usuario)

    if respuesta_ia:
        if not es_vision:
            with st.chat_message("assistant"):
                st.write(respuesta_ia)
                if responder_con_voz:
                    audio_out = texto_a_voz(respuesta_ia)
                    if audio_out: st.audio(audio_out, format="audio/mp3", autoplay=True)
        
        st.session_state.chat_history.append(AIMessage(content=respuesta_ia))
        if es_vision: st.rerun()