import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# --- CONFIGURACIÓN ---
st.set_page_config(page_icon="🥈", page_title="Asistente Silver Economy")

# --- 1. INTERRUPTOR DE SEGURIDAD (Para apagar el chat si gastas mucho) ---
if st.secrets.get("ESTADO_DEL_CHAT", "true") == "false":
    st.warning("🔒 Chat desactivado temporalmente por mantenimiento.")
    st.stop()

st.title("🥈 Asistente Silver Economy")

# --- 2. GESTIÓN DE API KEY (Desde Secretos) ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    # Si falla el secreto, pedimos manual (plan B)
    key = st.sidebar.text_input("API Key:", type="password")
    if not key:
        st.info("Ingresa la API Key para continuar.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = key

# --- 3. CARGA DE DATOS (A prueba de errores de ruta) ---
@st.cache_resource
def iniciar_base_datos():
    # TRUCO: Buscamos la carpeta Data basándonos en dónde está ESTE archivo app.py
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_data = os.path.join(ruta_base, "Data")
    
    if not os.path.exists(ruta_data):
        st.error(f"❌ No encuentro la carpeta Data en: {ruta_data}")
        return None
    
    docs = []
    with st.spinner("Leyendo documentos..."):
        for archivo in os.listdir(ruta_data):
            if archivo.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(ruta_data, archivo))
                docs.extend(loader.load())
    
    if not docs:
        st.warning("La carpeta Data existe pero no tiene PDFs.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return FAISS.from_documents(splits, OpenAIEmbeddings())

# Inicialización
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = iniciar_base_datos()

if st.session_state.vectorstore is None:
    st.stop() # Parar si no hay base de datos

st.session_state.retriever = st.session_state.vectorstore.as_retriever()

# --- 4. CHATBOT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
template = """Responde basado solo en el contexto: {context}
Historial: {chat_history}
Pregunta: {question}"""
prompt_temp = ChatPromptTemplate.from_template(template)

def responder(pregunta):
    docs = st.session_state.retriever.invoke(pregunta)
    contexto = "\n".join([d.page_content for d in docs])
    historial = "\n".join([f"{m.type}: {m.content}" for m in st.session_state.chat_history[-4:]])
    return llm.invoke(prompt_temp.format_messages(context=contexto, chat_history=historial, question=pregunta)).content

for msg in st.session_state.chat_history:
    st.chat_message(msg.type).write(msg.content)

if preg := st.chat_input("Pregunta aquí..."):
    st.session_state.chat_history.append(HumanMessage(content=preg))
    st.chat_message("user").write(preg)
    with st.chat_message("assistant"):
        resp = responder(preg)
        st.write(resp)
    st.session_state.chat_history.append(AIMessage(content=resp))