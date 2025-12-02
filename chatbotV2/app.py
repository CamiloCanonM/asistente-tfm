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
from langchain_core.documents import Document 
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_icon="🥈", page_title="KIVIA.AI", layout="centered")

# --- 🎨 CSS: ESTILO MODERNO Y LIMPIO ---
st.markdown("""
    <style>
    /* 1. Ocultar elementos de sistema de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* 2. Estilizar la cabecera */
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem !important;
        text-align: center;
    }
    
    /* 3. Botones más amigables (redondeados) */
    .stButton>button {
        border-radius: 20px;
        width: 100%;
    }
    
    /* 4. Ajustar el ancho del chat para que parezca una app móvil */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max_width: 700px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CAPTURA DE PARÁMETROS URL ---
params = st.query_params
usuario_nombre = params.get("nombre", "Usuario")
usuario_edad = params.get("edad", "No especificada")
usuario_peso = params.get("peso", "No especificado")
usuario_condicion = params.get("condicion", "Ninguna")

PERFIL_CLINICO = f"""
- Nombre: {usuario_nombre}
- Edad: {usuario_edad}
- Peso: {usuario_peso}
- Condición: {usuario_condicion}
"""

if st.secrets.get("ESTADO_DEL_CHAT", "true") == "false":
    st.warning("🔒 Mantenimiento.")
    st.stop()

# --- API KEY ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    key = st.sidebar.text_input("API Key:", type="password")
    if not key: st.stop()
    os.environ["OPENAI_API_KEY"] = key

client_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- FUNCIONES ---
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
                {"role": "user", "content": [{"type": "text", "text": "Describe detalladamente esta imagen."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ], max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e: return f"Error: {e}"

def leer_reloj_en_vivo():
    # 👇👇👇 TU LINK AQUÍ 👇👇👇
    url_sheet = "https://docs.google.com/spreadsheets/d/e/TU_CODIGO/pub?output=csv"
    try:
        if "TU_CODIGO" in url_sheet: return None 
        df = pd.read_csv(url_sheet)
        return df.iloc[-1] 
    except: return None

# --- DATABASE ---
@st.cache_resource
def iniciar_base_datos():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_data = os.path.join(ruta_base, "Data")
    if not os.path.exists(ruta_data): return None
    docs = []
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

if "vectorstore" not in st.session_state: st.session_state.vectorstore = iniciar_base_datos()

# Agregar archivo usuario
def agregar_archivo_usuario(uploaded_file):
    texto_extraido = ""
    nombre_archivo = uploaded_file.name
    if nombre_archivo.endswith(".pdf"):
        with open("temp.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        texto_extraido = "\n".join([d.page_content for d in docs])
    elif nombre_archivo.endswith(".txt"):
        texto_extraido = uploaded_file.read().decode("utf-8")
    elif nombre_archivo.endswith((".png", ".jpg", ".jpeg")):
        with st.spinner("👀 Leyendo imagen..."):
            texto_extraido = analizar_imagen(uploaded_file.getvalue())
            
    if texto_extraido:
        nuevo_doc = Document(page_content=texto_extraido, metadata={"source": f"Usuario: {nombre_archivo}"})
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents([nuevo_doc])
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
        else:
            st.session_state.vectorstore.add_documents(splits)
        return True
    return False

if st.session_state.vectorstore:
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
else:
    st.session_state.retriever = None

# --- CEREBROS ---
llm_seguridad = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt_seguridad = ChatPromptTemplate.from_template("Clasifica: 1. PELIGRO, 2. NEGATIVO, 3. NORMAL. Mensaje: {mensaje}")
def analizar_riesgo(mensaje):
    return (prompt_seguridad | llm_seguridad).invoke({"mensaje": mensaje}).content.strip().upper()

llm_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
template_chat = f"""Eres KIVIA. Hablas con {usuario_nombre}.
PERFIL: {PERFIL_CLINICO}
Contexto: {{context}}
Pregunta: {{question}}"""
prompt_chat = ChatPromptTemplate.from_template(template_chat)

def responder_rag(pregunta, nombre):
    if st.session_state.retriever:
        docs = st.session_state.retriever.invoke(pregunta)
        contexto = "\n".join([d.page_content for d in docs])
    else: contexto = "Sin datos."
    return (prompt_chat | llm_chat).invoke({"context": contexto, "question": pregunta, "nombre_usuario": nombre}).content

# --- INTERFAZ ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "ultimo_audio_id" not in st.session_state: st.session_state.ultimo_audio_id = None

# HEADER
st.title(f"🥈 KIVIA.AI")
st.markdown(f"**Hola, {usuario_nombre}** 👋")

# BARRA LATERAL (Solo para cosas secundarias)
with st.sidebar:
    st.header("⚙️ Panel de Control")
    archivo_subido = st.file_uploader("📂 Subir Receta/PDF", type=["pdf", "txt", "png", "jpg"])
    if archivo_subido:
        if "ultimo_archivo" not in st.session_state or st.session_state.ultimo_archivo != archivo_subido.name:
            if agregar_archivo_usuario(archivo_subido):
                st.success("✅ Memorizado")
                st.session_state.ultimo_archivo = archivo_subido.name
                st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    
    st.divider()
    if st.button("🔄 Sincronizar Reloj"):
        # Tu lógica de reloj aquí
        pass

# --- ZONA DE CHAT (CENTRAL) ---
for msg in st.session_state.chat_history:
    st.chat_message(msg.type).write(msg.content)

st.write("") # Espacio
st.write("") # Espacio

# --- BARRA DE HERRAMIENTAS (MODERNA) ---
st.divider()
col_cam, col_mic, col_txt = st.columns([1, 1, 0.1])

with col_cam:
    # Popover es más limpio que expander
    with st.popover("📸 Cámara", use_container_width=True):
        imagen_capturada = st.camera_input("Foto", label_visibility="collapsed")

with col_mic:
    audio_data = mic_recorder(start_prompt="🎙️ Hablar", stop_prompt="⏹️ Fin", key='recorder')

# INPUT TEXTO
texto_input = st.chat_input(f"Escribe aquí...")

# --- LÓGICA ---
prompt_usuario = None
respuesta_ia = None
es_vision = False
responder_con_voz = False

# 1. VISIÓN
if imagen_capturada:
    if "ultima_foto_proc" not in st.session_state: st.session_state.ultima_foto_proc = None
    if imagen_capturada.getvalue() != st.session_state.ultima_foto_proc:
        prompt_usuario = "📸 [Imagen de cámara]"
        with st.spinner("👁️ Analizando..."):
            respuesta_ia = analizar_imagen(imagen_capturada.getvalue())
        es_vision = True
        st.session_state.ultima_foto_proc = imagen_capturada.getvalue()

# 2. AUDIO
elif audio_data and audio_data['id'] != st.session_state.ultimo_audio_id:
    texto = transcribir_audio(audio_data['bytes'])
    if texto:
        prompt_usuario = texto
        responder_con_voz = True
        st.session_state.ultimo_audio_id = audio_data['id']

# 3. TEXTO
elif texto_input:
    prompt_usuario = texto_input

# PROCESAMIENTO
if prompt_usuario:
    st.session_state.chat_history.append(HumanMessage(content=prompt_usuario))
    if not es_vision: st.chat_message("user").write(f"🗣️ {prompt_usuario}" if responder_con_voz else prompt_usuario)

    if not es_vision and not respuesta_ia:
        with st.chat_message("assistant"):
            with st.spinner("..."):
                riesgo = analizar_riesgo(prompt_usuario)
                if "PELIGRO" in riesgo:
                    respuesta_ia = "🚨 EMERGENCIA 112"
                    st.error("Alerta")
                    responder_con_voz = False
                elif "NEGATIVO" in riesgo:
                    respuesta_ia = responder_rag(f"[TRISTE] {prompt_usuario}", usuario_nombre)
                else:
                    respuesta_ia = responder_rag(prompt_usuario, usuario_nombre)

    if respuesta_ia:
        if not es_vision:
            with st.chat_message("assistant"):
                st.write(respuesta_ia)
                if responder_con_voz:
                    audio_out = texto_a_voz(respuesta_ia)
                    if audio_out: st.audio(audio_out, format="audio/mp3", autoplay=True)
        
        st.session_state.chat_history.append(AIMessage(content=respuesta_ia))
        if es_vision: st.rerun()




