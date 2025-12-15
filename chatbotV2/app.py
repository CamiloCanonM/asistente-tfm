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
import streamlit as st
import streamlit as st

# --- 1. CONFIGURACIÓN DE PÁGINA ---
# ESTO DEBE IR SIEMPRE PRIMERO. Solo una vez.
st.set_page_config(
    page_title="KIVIA.AI",
    page_icon="logo.png", 
    layout="centered"
)

# --- 2. ESTILOS CSS ---
st.markdown("""
    <style>
    /* Ocultar elementos de sistema de Streamlit (Hamburguesa, footer, deploy) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Estilizar el título */
    h1 {
        color: #0E4F75; /* Azul oscuro corporativo */
        font-family: sans-serif;
        text-align: center;
    }
    
    /* Botones redondeados al 100% de ancho */
    .stButton>button {
        border-radius: 20px;
        width: 100%;
        border: 1px solid #0E4F75; /* Borde azul */
        background-color: white;
        color: #0E4F75;
    }
    
    /* Efecto al pasar el ratón por encima del botón */
    .stButton>button:hover {
        background-color: #0E4F75;
        color: white;
        border-color: #0E4F75;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. MOSTRAR EL LOGO Y TÍTULO ---
# Usamos columnas para centrar el logo (1 parte vacía, 2 partes logo, 1 parte vacía)
col1, col2, col3 = st.columns([1, 2, 1]) 

with col2:
    try:
        # Intentamos cargar el logo. Si no existe, no rompe la app.
        st.image("logo.png", use_container_width=True)
    except:
        st.warning("⚠️ No encuentro el archivo 'logo.png'. Asegúrate de subirlo a la carpeta.")

st.title("KIVIA.AI")


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
    # LINK HOJA DE CALCULO
    url_sheet = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5BW0ZT3Mp5Sd9DdpmAKqgPC-iZzrGyRIM7zV-_gcBTw8eR3SJAqklacU462M5QtB8qhVUG7Q38Hw_/pub?output=csv"
    
    try:
        # Validación
        if "TU_CODIGO" in url_sheet: return None 
        
        # Leemos el CSV directamente de Google
        df = pd.read_csv(url_sheet)
        
        # Devolvemos la última fila (el dato más reciente)
        return df.iloc[-1] 
    except Exception as e:
        # Si falla, imprimimos el error en la consola de Streamlit para que lo veas
        print(f"Error leyendo Google Sheet: {e}")
        return None

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

# PROMPT DE SEGURIDAD (VERSIÓN PERMISIVA)
template_seguridad = """Actúa como un filtro de seguridad lógico.
Analiza el siguiente mensaje y clasifícalo.

Regla de Oro: Si la frase es ambigua, corta o pregunta por ubicación ("dónde", "cuándo"), clasifícala SIEMPRE como NORMAL.

Categorías:
1. PELIGRO: ÚNICAMENTE si hay una declaración EXPLÍCITA de querer morir o matar.
   (Ej: "Me voy a suicidar", "Quiero acabar con mi vida ahora").

2. NEGATIVO: Expresiones claras de tristeza o soledad.
   (Ej: "Me siento muy solo", "Estoy llorando").

3. NORMAL: Todo lo demás.
   - Preguntas de seguimiento ("¿Y dónde quedan?", "¿Cuáles son?").
   - Preguntas de salud o curiosidad.
   - Cualquier cosa que no sea un riesgo vital obvio.

Mensaje del usuario: {mensaje}

Clasificación (PELIGRO / NEGATIVO / NORMAL):"""

prompt_seguridad = ChatPromptTemplate.from_template(template_seguridad)

def analizar_riesgo(mensaje):
    return (prompt_seguridad | llm_seguridad).invoke({"mensaje": mensaje}).content.strip().upper()

# Configuración del Chat Principal
llm_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# --- PROMPT---
template_chat = """Eres un asistente virtual experto en Silver Economy, diseñado para acompañar a personas mayores y sus familias.
Tu prioridad es ser útil, pero sobre todo CÁLIDO, PACIENTE y RESPETUOSO.

Sigue estas reglas estrictas para responder:

1. 👋 SALUDOS: Si el usuario te saluda (ej: "hola", "buenos días"), IGNORA el contexto de los documentos. Simplemente responde el saludo con amabilidad, preséntate y pregunta en qué puedes ayudar.
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

5. IDIOMA Y PERSONALIZACIÓN:
   * Responde en el idioma que el usuario pregunte.
   * Dirígete al usuario por su nombre: {nombre_usuario}.

6. REGLA DE ORO (MEMORIA):
   - Mira el "Historial de conversación" abajo.
   - Si ves que YA has saludado a {nombre_usuario} antes, NO vuelvas a decir "Hola" ni te presentes de nuevo.
   - Si el usuario te hace una pregunta de seguimiento (ej: "y cuáles son?"), responde DIRECTAMENTE a la pregunta.

7. 🧠 USO DEL CONTEXTO:
   - Usa la información de abajo para responder.
   - Si la respuesta NO está en el contexto, di: "Lo siento, no tengo esa información específica en mis documentos". ¡Pero NO te pongas a saludar para rellenar el silencio!
 
---
PERFIL: {PERFIL_CLINICO}

CONTEXTO RECUPERADO (Tus conocimientos):
{context}
---
HISTORIAL DE CONVERSACIÓN (Lo que ya hablamos):
{chat_history}
---

Pregunta actual: {question}
Respuesta (Sin repetir saludos):"""

prompt_chat = ChatPromptTemplate.from_template(template_chat)

# --- FUNCIÓN  ---
def responder_rag(pregunta, nombre):
    # A. Recuperar documentos (Contexto)
    if st.session_state.retriever:
        docs = st.session_state.retriever.invoke(pregunta)
        contexto = "\n".join([d.page_content for d in docs])
    else: 
        contexto = "No hay documentos cargados."
    
    # B. Recuperar historial (Memoria)
    historial_texto = "\n".join([f"{m.type}: {m.content}" for m in st.session_state.chat_history[-4:]])
    
    # C. Enviarlo todo al cerebro (AQUÍ ESTABA EL ERROR)
    # Debemos usar las MISMAS claves que pusiste entre llaves {} en el texto de arriba
    return (prompt_chat | llm_chat).invoke({
        "context": contexto,
        "question": pregunta,
        "nombre_usuario": nombre,      # Coincide con {nombre_usuario}
        "PERFIL_CLINICO": PERFIL_CLINICO, # Coincide con {PERFIL_CLINICO} (Antes tenías "perfil")
        "chat_history": historial_texto
    }).content


# --- INTERFAZ ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "ultimo_audio_id" not in st.session_state: st.session_state.ultimo_audio_id = None

# HEADER
st.title(f"🥈 KIVIA.AI")
st.markdown(f"**Hola, {usuario_nombre}** 👋")


# --- BARRA LATERAL 

with st.sidebar:
    st.header("⚙️ Panel de Control")
    
    # ---------------------------------------------------------
    #  ZONA DE ARCHIVOS
    # ---------------------------------------------------------
    st.subheader("📂 Mis Documentos")
    archivo_subido = st.file_uploader("Subir Receta o PDF", type=["pdf", "txt", "png", "jpg","xlsx"])
    
    if archivo_subido:
        # Verificamos si es un archivo nuevo para no procesarlo 2 veces
        if "ultimo_archivo" not in st.session_state or st.session_state.ultimo_archivo != archivo_subido.name:
            if agregar_archivo_usuario(archivo_subido):
                st.success(f"✅ {archivo_subido.name} guardado en memoria.")
                st.session_state.ultimo_archivo = archivo_subido.name
                # Recargamos el buscador para que incluya este archivo nuevo
                st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    
    st.divider()

    # ---------------------------------------------------------
    # ZONA DE RELOJ INTELIGENTE (DIAGNÓSTICO) ⌚
    # ---------------------------------------------------------
    st.header("⌚ Monitor Wearable")
    
    if st.button("🔄 Sincronizar Reloj"):
        datos = leer_reloj_en_vivo()
        
        if datos is not None:
            # Mostramos los datos crudos para ver si Google responde
            # st.write(datos) # Descomenta esto si quieres ver la fila entera
            
            try:
                # Intentamos leer las columnas siendo flexibles con mayúsculas/minúsculas
                ritmo_bruto = datos.get('Ritmo') or datos.get('ritmo') or datos.iloc[2]
                pasos_bruto = datos.get('Pasos') or datos.get('pasos') or datos.iloc[3]

                # Limpiamos los datos por si tienen texto (ej: "70 bpm")
                ritmo = int(str(ritmo_bruto).replace("bpm", "").strip())
                pasos = int(str(pasos_bruto).replace("pasos", "").strip())
                
                # Mostramos las métricas bonitas
                st.metric("❤️ Ritmo", f"{ritmo} bpm", delta=f"{ritmo-70}")
                st.metric("👣 Pasos", f"{pasos}")
                
                # Lógica de Alerta Roja
                if ritmo > 100:
                    st.session_state.iot_alert = f"ALERTA CRÍTICA: Ritmo {ritmo} bpm detectado."
                    st.error(f"⚠️ ANOMALÍA: {ritmo} bpm es peligroso.")
                else:
                    st.success("✅ Signos vitales normales")
                    # Borramos alerta si ya pasó el peligro
                    if "iot_alert" in st.session_state: del st.session_state.iot_alert
            
            except Exception as e:
                st.error("Error leyendo números del Excel.")
                st.caption(f"Detalle técnico: {e}")
        else:
            st.warning("No se pudo conectar con Google Sheets.")

# --- ZONA DE CHAT (CENTRAL) ---
for msg in st.session_state.chat_history:
    st.chat_message(msg.type).write(msg.content)

st.write("") # Espacio
st.write("") # Espacio

# --- BARRA DE HERRAMIENTAS---
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
                    respuesta_ia = "🚨 EMERGENCIA: Llama al 123. No estás solo."
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

























