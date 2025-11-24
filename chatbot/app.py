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

# --- 1. INTERRUPTOR DE MANTENIMIENTO ---
if st.secrets.get("ESTADO_DEL_CHAT", "true") == "false":
    st.warning("🔒 Chat en mantenimiento.")
    st.stop()

st.title("🥈 Asistente Silver Economy")

# --- 2. GESTIÓN API KEY ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    key = st.sidebar.text_input("API Key:", type="password")
    if not key:
        st.info("Ingresa la API Key.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = key

# --- 3. CARGA DE DATOS ---
@st.cache_resource
def iniciar_base_datos():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_data = os.path.join(ruta_base, "Data")
    
    if not os.path.exists(ruta_data):
        return None
    
    docs = []
    # Usamos un spinner para indicar carga
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

template_seguridad = """Analiza el siguiente mensaje del usuario y clasifícalo en una de estas 3 categorías.
Responde SOLO con la palabra clave:

1. PELIGRO: Si hay menciones de suicidio, autolesión, querer morir, violencia extrema o emergencias médicas.
2. NEGATIVO: Si hay tristeza, soledad, enojo o frustración, pero sin riesgo inmediato de vida.
3. NORMAL: Saludos, preguntas de información, curiosidad o agradecimientos.

Mensaje: {mensaje}
Categoría:"""

prompt_seguridad = ChatPromptTemplate.from_template(template_seguridad)

def analizar_riesgo(mensaje):
    chain = prompt_seguridad | llm_seguridad
    respuesta = chain.invoke({"mensaje": mensaje})
    return respuesta.content.strip().upper() # Devuelve PELIGRO, NEGATIVO o NORMAL

# --- 5. CEREBRO RESPONDEDOR (EL BIBLIOTECARIO) 📚 ---
llm_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# Prompt empático ajustado
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
   * Discúlpate con elegancia: "Lamento decirte que no tengo información específica sobre ese punto en mis documentos actuales, pero estoy aquí para ayudarte con cualquier otro tema del archivo.

Contexto: {context}
Historial: {chat_history}
Pregunta: {question}
Respuesta:"""
prompt_chat = ChatPromptTemplate.from_template(template_chat)

def responder_rag(pregunta):
    docs = st.session_state.retriever.invoke(pregunta)
    contexto = "\n".join([d.page_content for d in docs])
    historial = "\n".join([f"{m.type}: {m.content}" for m in st.session_state.chat_history[-4:]])
    chain = prompt_chat | llm_chat
    return chain.invoke({"context": contexto, "chat_history": historial, "question": pregunta}).content

# --- 6. INTERFAZ DE CHAT INTELIGENTE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg.type).write(msg.content)

if user_input := st.chat_input("Escribe aquí..."):
    # 1. Mostramos el mensaje del usuario
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando emociones..."):
            
            # PASO A: Análisis de Riesgo
            riesgo = analizar_riesgo(user_input)
            
            # --- SEMÁFORO DE ACCIÓN ---
            
            # 🔴 CASO ROJO: RIESGO DE VIDA
            if "PELIGRO" in riesgo:
                respuesta = """🚨 **Mensaje Importante** 🚨
                
                Siento mucho que estés pasando por un momento tan difícil. No estás solo/a.
                Por favor, busca ayuda profesional inmediatamente.
                
                📞 **Línea de la Vida (Ejemplo):** 800-911-2000
                🏥 **Emergencias:** 112 / 911
                
                Aunque soy una IA y quiero ayudarte, en situaciones de crisis necesitas contacto humano urgente."""
                st.error("Se ha detectado contenido de riesgo. Protocolo de emergencia activado.")
            
            # 🟡 CASO AMARILLO: TRISTEZA/EMOCIÓN (Pero seguro)
            elif "NEGATIVO" in riesgo:
                st.info("💡 Detecto que este tema es sensible para ti. Te respondo con cuidado:")
                respuesta = responder_rag(user_input)
                
            # 🟢 CASO VERDE: NORMAL
            else:
                respuesta = responder_rag(user_input)
            
            st.write(respuesta)
            
    st.session_state.chat_history.append(AIMessage(content=respuesta))