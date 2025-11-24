import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_icon="🥈", page_title="Asistente Silver Economy")

# --- 🛑 INTERRUPTOR DE APAGADO (CONTROL DE GASTOS) 🛑 ---
# Si en la nube configuramos esto como "false", la app se bloquea y no gasta dinero.
estado_chat = st.secrets.get("ESTADO_DEL_CHAT", "true")

if estado_chat == "false":
    st.title("🥈 Asistente Silver Economy")
    st.warning("🔒 El servicio está temporalmente desactivado por mantenimiento. Por favor, contacta al administrador.")
    st.stop()

st.title("🥈 Asistente Silver Economy")

# --- GESTIÓN DE LA API KEY (SEGURIDAD) ---
# 1. Intenta leer la clave desde los Secretos de la Nube (Lo que configurarás luego)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# 2. Si no hay secreto (estás en local), la pide en la barra lateral
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
    if not openai_api_key:
        st.info("Por favor, introduce una API Key para comenzar.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = openai_api_key

# --- MOTOR DE IA (RAG) ---
@st.cache_resource
def iniciar_base_datos():
    ruta_data = "./Data"
    if not os.path.exists(ruta_data):
        st.error("Carpeta 'Data' no encontrada.")
        return None
    
    docs = []
    with st.spinner("Procesando documentos..."):
        for archivo in os.listdir(ruta_data):
            if archivo.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(ruta_data, archivo))
                docs.extend(loader.load())
    
    if not docs:
        st.error("No hay PDFs en la carpeta Data.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return FAISS.from_documents(splits, OpenAIEmbeddings())

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = iniciar_base_datos()

if st.session_state.vectorstore is not None:
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    st.stop() # Detener si no hay base de datos

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """Eres un asistente experto en Silver Economy.
Contexto: {context}
Historial: {chat_history}
Pregunta: {question}
Respuesta:"""
prompt_template = ChatPromptTemplate.from_template(template)

def responder(pregunta, historial):
    docs = st.session_state.retriever.invoke(pregunta)
    contexto = "\n\n".join([d.page_content for d in docs])
    historial_txt = "\n".join([f"{msg.type}: {msg.content}" for msg in historial[-4:]])
    prompt = prompt_template.format_messages(context=contexto, chat_history=historial_txt, question=pregunta)
    return llm.invoke(prompt).content

# --- INTERFAZ DE CHAT ---
for msg in st.session_state.chat_history:
    st.chat_message("assistant" if isinstance(msg, AIMessage) else "user").markdown(msg.content)

if user_input := st.chat_input("Haz tu pregunta sobre los documentos..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            response = responder(user_input, st.session_state.chat_history)
            st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))