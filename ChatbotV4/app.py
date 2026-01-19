import streamlit as st
import os
import io
import pandas as pd
import base64
import time
import sys
import pickle
import numpy as np

# --- LIBRERÍAS DE IA Y LANGCHAIN ---
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document 
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

# --- MÓDULOS PROPIOS (Tu arquitectura modular) ---
# Aseguramos que Python encuentre tus archivos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import social_view      # Mapas
import docs_view        # Gestor de archivos
import apariencia       # Estilos CSS
import planificador_iu  # La vista del cuestionario nuevo
# kivia_backend se usa dentro de planificador_iu, no hace falta importarlo aquí directamente

# ==========================================
# CONFIGURACIÓN DE PÁGINA (SIEMPRE AL INICIO)
# ==========================================
st.set_page_config(page_title="KIVIA.AI", page_icon="🧬", layout="wide")

# ==========================================
# LÓGICA DEL CHATBOT (Encapsulada para orden)
# ==========================================
def vista_chatbot():
    
    # 1. CARGAR ESTILOS
    try:
        apariencia.cargar_estilos_css()
        apariencia.mostrar_header()
    except:
        st.markdown("### Kivia Health AI") # Fallback si falla apariencia

    # 2. CAPTURA DE PARÁMETROS URL Y PERFIL
    params = st.query_params
    usuario_nombre = params.get("nombre", "Usuario")
    usuario_edad = params.get("edad", "No especificada")
    usuario_peso = params.get("peso", "No especificado")
    usuario_condicion = params.get("condicion", "Ninguna")

    # --- 3. RECUPERAR DATOS DEL PLANIFICADOR (MEMORIA COMPARTIDA) ---
    info_planificador = ""
    if 'kivia_data' in st.session_state:
        datos = st.session_state['kivia_data']
        # Convertimos el diccionario de perfil en texto legible
        detalles_perfil = datos.get('perfil', {})
        texto_perfil = ", ".join([f"{k}: {v}" for k,v in detalles_perfil.items()])
        
        info_planificador = f"""
        [RESULTADOS RECIENTES DEL PLANIFICADOR DE HÁBITOS]
        - Puntaje Kivia: {datos.get('score', 'N/A')}/100
        - Probabilidad de Éxito: {datos.get('prob', 0)*100}%
        - Detalles del usuario: {texto_perfil}
        
        INSTRUCCIÓN: Si el usuario pregunta por su salud o hábitos, usa estos datos para darle consejos personalizados.
        Felicítalo si el puntaje es alto (>70), o anímalo si es bajo.
        """

    PERFIL_CLINICO = f"""
    - Nombre: {usuario_nombre}
    - Edad: {usuario_edad}
    - Peso: {usuario_peso}
    - Condición: {usuario_condicion}

    {info_planificador} 
    """

    # 4. VERIFICACIONES DE SEGURIDAD Y API
    if st.secrets.get("ESTADO_DEL_CHAT", "true") == "false":
        st.warning("🔒 Sistema en mantenimiento.")
        st.stop()

    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        key = st.sidebar.text_input("OpenAI API Key:", type="password")
        if not key:
            st.info("Por favor ingresa la API Key para continuar.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = key

    client_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 5. FUNCIONES AUXILIARES (Audio, Imagen, Reloj)
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
                    {"role": "user", "content": [{"type": "text", "text": "Describe detalladamente esta imagen médica o medicamento. Sé preciso."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ], max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e: return f"Error visual: {e}"

    def leer_reloj_en_vivo():
        url_sheet = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5BW0ZT3Mp5Sd9DdpmAKqgPC-iZzrGyRIM7zV-_gcBTw8eR3SJAqklacU462M5QtB8qhVUG7Q38Hw_/pub?output=csv"
        try:
            if "TU_CODIGO" in url_sheet: return None 
            df = pd.read_csv(url_sheet)
            return df.iloc[-1] 
        except Exception as e:
            print(f"Error reloj: {e}")
            return None

    # 6. BASE DE DATOS VECTORIAL (RAG)
    @st.cache_resource
    def iniciar_base_datos():
        ruta_base = os.path.dirname(os.path.abspath(__file__))
        ruta_data = os.path.join(ruta_base, "Data") # Asegúrate que esta carpeta exista
        if not os.path.exists(ruta_data): 
            os.makedirs(ruta_data, exist_ok=True)
            return None
            
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

    def agregar_archivo_usuario(uploaded_file):
        texto_extraido = ""
        nombre_archivo = uploaded_file.name
        try:
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
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")
        return False

    if st.session_state.vectorstore:
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    else:
        st.session_state.retriever = None

    # 7. CONFIGURACIÓN DEL CEREBRO (LLM)
    llm_seguridad = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    template_seguridad = """Actúa como un filtro de seguridad.
    Clasifica el mensaje:
    1. PELIGRO: Solo si hay riesgo inminente de muerte o suicidio explícito.
    2. NEGATIVO: Tristeza profunda o soledad.
    3. NORMAL: Todo lo demás (incluyendo preguntas médicas, ubicación, saludos).
    Mensaje: {mensaje}
    Clasificación:"""
    
    prompt_seguridad = ChatPromptTemplate.from_template(template_seguridad)

    def analizar_riesgo(mensaje):
        return (prompt_seguridad | llm_seguridad).invoke({"mensaje": mensaje}).content.strip().upper()

    template_chat = """Eres KIVIA, un asistente experto en salud y Silver Economy.
    
    CONTEXTO DEL USUARIO:
    {PERFIL_CLINICO}

    CONOCIMIENTO (Base de datos):
    {context}

    UBICACIÓN (GPS):
    {geo_contexto}

    HISTORIAL:
    {chat_history}

    INSTRUCCIONES:
    1. Responde de forma cálida y empática.
    2. Usa el CONOCIMIENTO para responder. Si no lo sabes, dilo.
    3. Si hay datos del PLANIFICADOR, úsalos para personalizar la respuesta.
    4. Si preguntan "dónde", usa la UBICACIÓN.

    Usuario: {question}
    Kivia:"""

    prompt_chat = ChatPromptTemplate.from_template(template_chat)

    def responder_rag(pregunta, nombre):
        if st.session_state.retriever:
            docs = st.session_state.retriever.invoke(pregunta)
            contexto = "\n".join([d.page_content for d in docs])
        else: 
            contexto = "No hay documentos base cargados."
        
        historial_texto = "\n".join([f"{m.type}: {m.content}" for m in st.session_state.chat_history[-4:]])
        geo_data = st.session_state.get("geo_contexto", "No se han buscado lugares cercanos.")

        return (prompt_chat | llm_chat).invoke({
            "context": contexto,
            "geo_contexto": geo_data,
            "question": pregunta,
            "nombre_usuario": nombre,
            "PERFIL_CLINICO": PERFIL_CLINICO,
            "chat_history": historial_texto
        }).content

    # 8. INTERFAZ GRÁFICA DEL CHAT
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "ultimo_audio_id" not in st.session_state: st.session_state.ultimo_audio_id = None
    if "mostrar_camara" not in st.session_state: st.session_state.mostrar_camara = False

    st.markdown(f"**Hola, {usuario_nombre}** 👋")

    # --- SIDEBAR ESPECÍFICO DEL CHAT ---
    with st.sidebar:
        # A. Mapas
        try:
            social_view.renderizar_sidebar()
        except: st.warning("Módulo de mapas no cargado")
        
        st.divider()
        
        # B. Documentos
        try:
            docs_view.renderizar_documentos()
            
            # Lógica de carga automática de documentos subidos
            CARPETA_DOCS = "base_conocimiento"
            if "archivos_leidos" not in st.session_state: st.session_state.archivos_leidos = set()
            
            if os.path.exists(CARPETA_DOCS):
                for nombre_archivo in os.listdir(CARPETA_DOCS):
                    if nombre_archivo not in st.session_state.archivos_leidos:
                        ruta_completa = os.path.join(CARPETA_DOCS, nombre_archivo)
                        try:
                            with open(ruta_completa, "rb") as f:
                                contenido = f.read()
                                archivo_simulado = io.BytesIO(contenido)
                                archivo_simulado.name = nombre_archivo
                                if agregar_archivo_usuario(archivo_simulado):
                                    st.session_state.archivos_leidos.add(nombre_archivo)
                                    st.toast(f"✅ Aprendido: {nombre_archivo}")
                        except: pass
            
            # Actualizar retriever si hubo cambios
            if st.session_state.vectorstore:
                st.session_state.retriever = st.session_state.vectorstore.as_retriever()

        except: st.warning("Módulo de documentos no disponible")

        st.divider()

        # C. Reloj
        st.header("⌚ Wearable")
        if st.button("🔄 Sincronizar Reloj"):
            datos_reloj = leer_reloj_en_vivo()
            if datos_reloj is not None:
                try:
                    ritmo = int(str(datos_reloj.iloc[2]).replace("bpm","").strip())
                    st.metric("Ritmo", f"{ritmo} bpm")
                    if ritmo > 100: st.error("⚠️ Taquicardia detectada")
                    else: st.success("✅ Normal")
                except: st.error("Error formato reloj")
            else: st.warning("Reloj desconectado")

    # --- ZONA CENTRAL DE CHAT ---
    
    # Historial
    for msg in st.session_state.chat_history:
        st.chat_message(msg.type).write(msg.content)
    
    st.write("---")
    
    # Botonera
    c1, col_camara, col_mic, c4 = st.columns([1, 2, 2, 1])
    with col_camara:
        if st.button("📷 Cámara", use_container_width=True, key="btn_cam"):
            st.session_state.mostrar_camara = not st.session_state.mostrar_camara
    with col_mic:
        audio_data = mic_recorder(start_prompt="🎙️ Hablar", stop_prompt="⏹️ Enviar", just_once=True, key="grabadora")

    # Inputs
    imagen_capturada = None
    if st.session_state.mostrar_camara:
        st.write("### 📸 Tomar Foto")
        imagen_capturada = st.camera_input("Enfoca", label_visibility="collapsed")
    
    texto_input = st.chat_input("Escribe tu consulta...")

    # Procesamiento
    prompt_usuario = None
    es_vision = False
    respuesta_ia = None
    responder_con_voz = False

    if imagen_capturada:
        if "ultima_foto_proc" not in st.session_state: st.session_state.ultima_foto_proc = None
        if imagen_capturada.getvalue() != st.session_state.ultima_foto_proc:
            prompt_usuario = "📸 [Imagen enviada]"
            es_vision = True
            st.session_state.ultima_foto_proc = imagen_capturada.getvalue()
            with st.spinner("Analizando imagen..."):
                respuesta_ia = analizar_imagen(imagen_capturada.getvalue())

    elif audio_data and audio_data['id'] != st.session_state.ultimo_audio_id:
        st.session_state.ultimo_audio_id = audio_data['id']
        texto_transcrito = transcribir_audio(audio_data['bytes'])
        if texto_transcrito:
            prompt_usuario = texto_transcrito
            responder_con_voz = True

    elif texto_input:
        prompt_usuario = texto_input

    # Respuesta
    if prompt_usuario:
        st.session_state.chat_history.append(HumanMessage(content=prompt_usuario))
        if not es_vision:
            st.chat_message("user").write(prompt_usuario)
        
        if not respuesta_ia:
            with st.spinner("Pensando..."):
                riesgo = analizar_riesgo(prompt_usuario)
                if "PELIGRO" in riesgo:
                    respuesta_ia = "🚨 ALERTA: Detecto riesgo grave. Por favor llama a emergencias."
                    responder_con_voz = True
                else:
                    respuesta_ia = responder_rag(prompt_usuario, usuario_nombre)
        
        st.session_state.chat_history.append(AIMessage(content=respuesta_ia))
        st.chat_message("assistant").write(respuesta_ia)
        
        if responder_con_voz:
            audio_out = texto_a_voz(respuesta_ia)
            if audio_out: st.audio(audio_out, format="audio/mp3", autoplay=True)
        
        if es_vision: st.rerun()

    # Mapas al final
    try:
        st.divider()
        social_view.mostrar_mapa_central()
    except: pass


# ==========================================
# FUNCIÓN PRINCIPAL (MAIN ROUTER)
# ==========================================
def main():
    # --- MENÚ LATERAL PRINCIPAL ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100)
        st.title("Kivia Health")
        
        # El Selector Maestro
        menu = st.radio("Navegación", ["🤖 Chatbot IA", "📊 Planificador de Hábitos"])
        
        st.divider()
        st.info("Sistema v2.0 - Modular")

    # --- ENRUTAMIENTO ---
    
    if menu == "📊 Planificador de Hábitos":
        # Llamada limpia al módulo que creamos
        planificador_iu.renderizar_planificador()

    elif menu == "🤖 Chatbot IA":
        # Llamada a la función encapsulada del chat
        vista_chatbot()

# PUNTO DE ENTRADA
if __name__ == "__main__":
    main()







