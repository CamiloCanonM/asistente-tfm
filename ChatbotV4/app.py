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
import os 
import social_view
import docs_view
import apariencia
import sys
import pickle
import numpy as np

#
st.set_page_config(page_title="KIVIA.AI", page_icon="🧬", layout="wide")

# Parche para encontrar archivos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==========================================
# BLOQUE A: PLANIFICADOR (Lógica Frontend Flask + Backend Original)
# ==========================================
import numpy as np
import pickle
import os

# --- 1. CLASES ORIGINALES (Indispensables para el pickle) ---
class ModelConfig:
    def __init__(self):
        self.n_components = 20
        self.pca_variance_threshold = 0.85
        self.xgb_auc_target = 0.80
        self.recommendation_accuracy_target = 0.75
        self.regression_r2_target = 0.70

class HabitModel:
    def __init__(self):
        self.config = ModelConfig()
        self.pca = None
        self.scaler = None
        self.xgb_model = None       
        self.regression_model = None 
        self.trained = False

# --- 2. CARGA DEL CEREBRO ---
@st.cache_resource
def cargar_cerebro_completo():
    ruta_app = os.path.dirname(os.path.abspath(__file__))
    rutas = [os.path.join(ruta_app, "habit_model.pkl"), os.path.join(ruta_app, "models", "habit_model.pkl")]
    for ruta in rutas:
        if os.path.exists(ruta):
            try:
                with open(ruta, "rb") as f:
                    return pickle.load(f)
            except: continue
    return None

# --- 3. LÓGICA DE TRADUCCIÓN (Extraída de app_flask.py) ---
def procesar_cuestionario_inteligente(respuestas):
    """
    Convierte las respuestas humanas en el vector de 50 características
    usando la lógica de tu Frontend original.
    """
    # 1. Crear vector base de 50 ceros
    features = np.zeros((1, 50))
    
    # 2. Extracción de valores clave (Normalizados 0.0 a 1.0)
    energia = respuestas.get("energia", 0.5)
    sueño = respuestas.get("sueño", 0.5)
    estres = respuestas.get("estres", 0.5)
    ejercicio = respuestas.get("ejercicio", 0.0)
    animo = respuestas.get("animo", 0.5)
    disciplina = respuestas.get("disciplina", 0.5)
    
    # 3. Mapeo directo a las primeras posiciones (Core Features)
    # Según tu lógica original, las primeras columnas son las biológicas
    features[0, 0] = energia
    features[0, 1] = sueño
    features[0, 2] = estres
    features[0, 3] = ejercicio
    features[0, 4] = animo
    features[0, 5] = disciplina
    
    # 4. Lógica de Relleno Inteligente (Simulación de app_flask.py)
    # Tu frontend usaba promedios para rellenar las características latentes (40-50)
    promedio_general = (energia + sueño + (1-estres) + ejercicio + animo) / 5
    
    # Rellenamos bloques del vector con patrones lógicos en lugar de ceros
    # Bloque 10-20: Relacionado con consistencia (basado en disciplina)
    features[0, 10:20] = disciplina * 0.8 + np.random.normal(0, 0.05, 10)
    
    # Bloque 20-30: Relacionado con bienestar (basado en sueño y estrés inverso)
    bienestar = (sueño + (1-estres)) / 2
    features[0, 20:30] = bienestar + np.random.normal(0, 0.05, 10)
    
    # Bloque 42-45: Índices específicos que tu app_flask calculaba como promedios
    features[0, 42] = promedio_general
    features[0, 43] = promedio_general * 1.1  # Proyección futura
    features[0, 44] = promedio_general * 0.9  # Estado base
    
    return features

# --- 4. INTERFAZ DE USUARIO ---
def renderizar_planificador_interno():
    st.title("🧠 Diagnóstico Profundo Kivia")
    st.markdown("Responde este cuestionario para que la IA analice tus patrones ocultos.")

    with st.form("cuestionario_kivia"):
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("⚡ Estado Físico")
            energia_in = st.select_slider("Nivel de Energía diario", 
                                        options=["Muy bajo", "Bajo", "Moderado", "Alto", "Muy Alto"], value="Moderado")
            ejercicio_in = st.select_slider("Frecuencia de Ejercicio", 
                                          options=["Nunca", "1 día/sem", "3 días/sem", "5+ días/sem"], value="1 día/sem")
            sueño_in = st.slider("Calidad de Sueño (0-100)", 0, 100, 70)

        with c2:
            st.subheader("🧘 Estado Mental")
            estres_in = st.select_slider("Nivel de Estrés", 
                                       options=["Zen", "Bajo", "Manejable", "Alto", "Crítico"], value="Manejable")
            animo_in = st.slider("Estado de Ánimo General (0-100)", 0, 100, 60)
            disciplina_in = st.select_slider("Autodisciplina percibida", 
                                           options=["Baja", "Variable", "Alta", "Hierro"], value="Variable")

        # Botón de envío
        submitted = st.form_submit_button("🚀 Analizar mis Probabilidades", type="primary")

    if submitted:
        modelo = cargar_cerebro_completo()
        
        if modelo:
            try:
                # --- A. TRADUCCIÓN DE TEXTO A NÚMEROS (0.0 - 1.0) ---
                map_energia = {"Muy bajo": 0.1, "Bajo": 0.3, "Moderado": 0.5, "Alto": 0.8, "Muy Alto": 1.0}
                map_ejer = {"Nunca": 0.0, "1 día/sem": 0.3, "3 días/sem": 0.7, "5+ días/sem": 1.0}
                map_estres = {"Zen": 0.0, "Bajo": 0.2, "Manejable": 0.5, "Alto": 0.8, "Crítico": 1.0}
                map_disci = {"Baja": 0.2, "Variable": 0.5, "Alta": 0.8, "Hierro": 1.0}

                respuestas_dict = {
                    "energia": map_energia[energia_in],
                    "ejercicio": map_ejer[ejercicio_in],
                    "sueño": sueño_in / 100.0,
                    "estres": map_estres[estres_in],
                    "animo": animo_in / 100.0,
                    "disciplina": map_disci[disciplina_in]
                }

                # --- B. MAGIA: GENERAR VECTOR DE 50 CARACTERÍSTICAS ---
                # Usamos la función inteligente que imita a tu app_flask.py
                vector_50 = procesar_cuestionario_inteligente(respuestas_dict)

                # --- C. PREDICCIÓN (Scaler -> PCA -> XGB/Regresión) ---
                datos_escalados = modelo.scaler.transform(vector_50)
                datos_pca = modelo.pca.transform(datos_escalados)
                
                # Predicciones
                prob_exito = modelo.xgb_model.predict_proba(datos_pca)[0, 1]
                score_raw = modelo.regression_model.predict(datos_pca)[0]
                kivia_score = int(max(0, min(100, score_raw)))

                # --- D. RESULTADOS ---
                st.session_state['kivia_data'] = {
                    "score": kivia_score, 
                    "prob": round(prob_exito, 2),
                    "perfil": respuestas_dict # Guardamos para el chatbot
                }
                
                # Layout de Resultados
                st.divider()
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    # Gráfico de dona simple con metric
                    st.metric("Kivia Score", f"{kivia_score}/100", delta=f"{'Positivo' if kivia_score > 60 else 'Mejorable'}")
                    st.progress(kivia_score / 100)
                
                with col_res2:
                    if prob_exito > 0.7:
                        st.success(f"🌟 **Alta Probabilidad de Éxito ({prob_exito:.0%})**\n\nTu perfil actual sugiere que tienes la energía y mentalidad correctas para adoptar nuevos hábitos.")
                    elif prob_exito > 0.4:
                        st.warning(f"⚖️ **Probabilidad Moderada ({prob_exito:.0%})**\n\nEstás en un punto de equilibrio. Pequeños ajustes en tu estrés o sueño podrían disparar tu éxito.")
                    else:
                        st.error(f"🛡️ **Probabilidad Baja ({prob_exito:.0%})**\n\nEl sistema detecta resistencia. Es mejor empezar con hábitos muy pequeños (micro-hábitos) para no saturarte.")

            except Exception as e:
                st.error(f"Error en el análisis: {e}")
                st.info("Intenta recargar la página.")
        else:
            st.error("❌ No se encontró el modelo. Verifica que 'habit_model.pkl' esté subido.")

# ==========================================
# BLOQUE A: LÓGICA DEL PLANIFICADOR (ORIGINAL ADAPTADA)
# ==========================================
import numpy as np
import pandas as pd
import pickle
import os

# --- 1. DEFINICIÓN DE CLASES ORIGINALES ---
# Necesarias para que 'pickle' reconozca la estructura del archivo guardado

class ModelConfig:
    def __init__(self):
        self.n_components = 20
        self.pca_variance_threshold = 0.85
        self.xgb_auc_target = 0.80
        self.recommendation_accuracy_target = 0.75
        self.regression_r2_target = 0.70

class HabitModel:
    def __init__(self):
        self.config = ModelConfig()
        self.pca = None
        self.scaler = None
        self.xgb_model = None       # Clasificador (Probabilidad)
        self.regression_model = None # Regresión (Score 0-100)
        self.trained = False

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def cargar_cerebro_planificador():
    # Buscamos el archivo en la misma carpeta
    ruta_app = os.path.dirname(os.path.abspath(__file__))
    rutas = [
        os.path.join(ruta_app, "habit_model.pkl"),
        os.path.join(ruta_app, "models", "habit_model.pkl")
    ]
    
    ruta_final = next((r for r in rutas if os.path.exists(r)), None)
    
    if ruta_final:
        try:
            with open(ruta_final, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error cargando el cerebro: {e}")
            return None
    return None

# --- 3. INTERFAZ GRÁFICA ---
def renderizar_planificador_interno():
    st.title("📊 Planificador de Hábitos (Motor Original)")
    st.markdown("Este módulo utiliza tu algoritmo de **XGBoost + PCA** original.")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        
        # --- INPUTS ---
        # Estos son los datos visuales, pero internamente necesitamos 50 variables
        energia = c1.slider("Nivel de Energía (0-100)", 0, 100, 50)
        sueno = c2.slider("Calidad de Sueño (0-100)", 0, 100, 50)
        estres_txt = c1.select_slider("Nivel de Estrés", ["Bajo", "Medio", "Alto"])
        ejercicio_txt = c2.select_slider("Ejercicio Semanal", ["Nada", "Poco", "Mucho"])
        
        if st.button("🚀 Ejecutar Análisis Completo", type="primary", use_container_width=True):
            
            modelo = cargar_cerebro_planificador()
            
            if modelo:
                try:
                    # --- RECONSTRUCCIÓN DEL VECTOR DE 50 CARACTERÍSTICAS ---
                    # Tu modelo original espera 50 datos. Como aquí solo pedimos 4,
                    # simulamos el resto para que la matemática funcione.
                    
                    # 1. Mapeo básico
                    mapa = {"Bajo": 0, "Medio": 1, "Alto": 2, "Nada": 0, "Poco": 1, "Mucho": 2}
                    
                    # 2. Generamos un vector base de 50 ceros (o aleatorio controlado)
                    # Usamos random.randn como en tu 'modelo_v2.py' para simular variabilidad biológica
                    input_vector = np.random.randn(1, 50)
                    
                    # 3. Inyectamos los datos reales del usuario en las primeras posiciones
                    # (Esto asume que las primeras columnas son las más importantes, o simplemente
                    #  modula el vector aleatorio con la realidad del usuario)
                    input_vector[0, 0] = energia / 100.0  # Normalizado
                    input_vector[0, 1] = sueno / 100.0
                    input_vector[0, 2] = mapa[estres_txt]
                    input_vector[0, 3] = mapa[ejercicio_txt]
                    
                    # --- PROCESAMIENTO EXACTO DE TU MODELO ORIGINAL ---
                    # 1. Scaler
                    datos_escalados = modelo.scaler.transform(input_vector)
                    
                    # 2. PCA
                    datos_pca = modelo.pca.transform(datos_escalados)
                    
                    # 3. Predicción (Usamos ambas partes de tu cerebro: XGB y LinearRegression)
                    prob_exito = modelo.xgb_model.predict_proba(datos_pca)[0, 1]
                    kivia_score_raw = modelo.regression_model.predict(datos_pca)[0]
                    
                    # Limpieza del score (0 a 100)
                    kivia_score = int(max(0, min(100, kivia_score_raw)))
                    
                    # --- RESULTADOS ---
                    st.session_state['kivia_data'] = {
                        "score": kivia_score, 
                        "prob": round(prob_exito, 2)
                    }
                    
                    # Mostrar métricas visuales
                    col_res1, col_res2 = st.columns(2)
                    col_res1.metric("Kivia Score", f"{kivia_score}/100")
                    col_res2.metric("Probabilidad Adopción", f"{prob_exito:.1%}")
                    
                    # Interpretación (Copiada de tu 'servicio.py')
                    if prob_exito >= 0.8:
                        msg = "🌟 Muy alta probabilidad de éxito."
                    elif prob_exito >= 0.6:
                        msg = "✅ Alta probabilidad de éxito."
                    else:
                        msg = "⚠️ Probabilidad baja - requiere apoyo."
                    
                    st.info(msg)
                    st.success("Los datos han sido enviados al Chatbot para que te aconseje.")

                except Exception as e:
                    st.error(f"Error en el motor de inferencia: {e}")
                    st.caption("Detalle: Verifica que 'HabitModel' tenga cargados scaler, pca, xgb_model y regression_model.")
            else:
                st.error("❌ No se encontró el archivo 'habit_model.pkl'.")

# ==========================================
# BLOQUE B: TU CÓDIGO CHATBOT
# ==========================================

def vista_chatbot():
    
    #CARGAR LA APARIENCIA
    apariencia.cargar_estilos_css()  # Inyecta el CSS
    apariencia.mostrar_header()      # Muestra el Logo
    
    
    # --- CAPTURA DE PARÁMETROS URL ---
    params = st.query_params
    usuario_nombre = params.get("nombre", "Usuario")
    usuario_edad = params.get("edad", "No especificada")
    usuario_peso = params.get("peso", "No especificado")
    usuario_condicion = params.get("condicion", "Ninguna")


    #  ---RECUPERAR DATOS DEL PLANIFICADOR
   
    info_planificador = ""
    if 'kivia_data' in st.session_state:
        datos = st.session_state['kivia_data']
        info_planificador = f"""
        RESULTADOS DEL PLANIFICADOR DE HÁBITOS:
        - Puntaje de Salud: {datos.get('score', 'N/A')}/100
        - Probabilidad de Adherencia: {datos.get('prob', 'N/A')}
        (Usa estos datos para felicitar o aconsejar al usuario si pregunta por su análisis).
        """
    # =======================================================
    
    PERFIL_CLINICO = f"""
    - Nombre: {usuario_nombre}
    - Edad: {usuario_edad}
    - Peso: {usuario_peso}
    - Condición: {usuario_condicion}

    {info_planificador}  <-- ¡AQUÍ INYECTAMOS LOS DATOS!
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
    llm_seguridad = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
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
    llm_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
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
    
    8. IMPORTANTE: Si el usuario pregunta "dónde" o por ubicación, usa los "LUGARES CERCANOS" para recomendar sitios específicos.
     
    ---
    PERFIL: {PERFIL_CLINICO}
    
    CONTEXTO RECUPERADO (Tus conocimientos):
    {context}
    
    LUGARES CERCANOS (Detectados por GPS):
    {geo_contexto}
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
        
        # B. Recuperar Historial
        historial_texto = "\n".join([f"{m.type}: {m.content}" for m in st.session_state.chat_history[-4:]])
        
        # C. Recuperar Contexto GEO (El puente con el mapa)
        # Si no existe la variable, ponemos un texto vacío para que no falle
        geo_data = st.session_state.get("geo_contexto", "No se han buscado lugares cercanos.")
    
        # D. ENVIAR AL CEREBRO
        # ¡IMPORTANTE! Aquí enviamos TODAS las variables que tu prompt pide:
        return (prompt_chat | llm_chat).invoke({
            "context": contexto,
            "geo_contexto": geo_data,        # <--- Para {geo_contexto}
            "question": pregunta,            # <--- Para {question}
            "nombre_usuario": nombre,        # <--- Para {nombre_usuario} (¡No lo borres!)
            "PERFIL_CLINICO": PERFIL_CLINICO,# <--- Para {PERFIL_CLINICO}
            "chat_history": historial_texto  # <--- Para {chat_history}
        }).content
    
    
    # --- INTERFAZ ---
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "ultimo_audio_id" not in st.session_state: st.session_state.ultimo_audio_id = None
    
    # HEADER
    st.markdown(f"**Hola, {usuario_nombre}** 👋")
    
    
    # --- BARRA LATERAL 
    
    with st.sidebar:
            
        # ---------------------------------------------------------
        # 1. INTEGRACIÓN SOCIAL VIEW (MAPAS)
        # ---------------------------------------------------------
        social_view.renderizar_sidebar()
        # Solo llamamos a la función. Ella se encarga de todo.
        
        st.divider()
        
        
        # ---------------------------------------------------------
        #  ZONA DE ARCHIVOS
        # ---------------------------------------------------------
        # 1. MOSTRAR EL GESTOR DE ARCHIVOS (VISUAL)
        # Esto muestra los botones y guarda los archivos en la carpeta
        docs_view.renderizar_documentos()
        # 2. PUENTE DE MEMORIA (CORREGIDO)
        CARPETA_DOCS = "base_conocimiento"
        
        # Inicializamos la memoria de qué archivos ya leyó la IA
        if "archivos_leidos" not in st.session_state:
            st.session_state.archivos_leidos = set()
    
        if os.path.exists(CARPETA_DOCS):
            archivos_en_disco = os.listdir(CARPETA_DOCS)
            
            for nombre_archivo in archivos_en_disco:
                # Si hay un archivo en la carpeta que la IA no ha leído aún...
                if nombre_archivo not in st.session_state.archivos_leidos:
                    
                    ruta_completa = os.path.join(CARPETA_DOCS, nombre_archivo)
                    
                    try:
                        # LEEMOS EL ARCHIVO DEL DISCO
                        with open(ruta_completa, "rb") as f:
                            contenido_bytes = f.read()
                        
                        # --- EL TRUCO PARA QUE NO DE ERROR ---
                        # Creamos un archivo "virtual" en memoria que sí permite cambiar el nombre
                        archivo_simulado = io.BytesIO(contenido_bytes)
                        archivo_simulado.name = nombre_archivo 
                        
                        with st.spinner(f"🧠 Aprendiendo: {nombre_archivo}..."):
                            # LLAMAMOS A TU FUNCIÓN ORIGINAL CON EL ARCHIVO SIMULADO
                            exito = agregar_archivo_usuario(archivo_simulado)
                            
                            if exito:
                                st.session_state.archivos_leidos.add(nombre_archivo)
                                st.toast(f"✅ Memoria actualizada con: {nombre_archivo}")
                    except Exception as e:
                        st.error(f"Error leyendo {nombre_archivo}: {e}")
    
        # Si hubo cambios, actualizamos el buscador
        if len(archivos_en_disco) > 0:
            if "vectorstore" in st.session_state:
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
    
    # ==========================================
    # 1. PREPARACIÓN (Variables y Estado)
    # ==========================================
    # Inicializamos variables críticas para evitar errores (AttributeError)
    if "mostrar_camara" not in st.session_state:
        st.session_state.mostrar_camara = False
    if "ultimo_audio_id" not in st.session_state:
        st.session_state.ultimo_audio_id = None
    
    # Variables temporales para esta vuelta del bucle
    imagen_capturada = None
    prompt_usuario = None
    es_vision = False
    respuesta_ia = None
    responder_con_voz = False
    
    
    # ==========================================
    # 2. MOSTRAR HISTORIAL DE CHAT
    # ==========================================
    for msg in st.session_state.chat_history:
        st.chat_message(msg.type).write(msg.content)
    
    st.write("") # Espacio visual
    
    
    # ==========================================
    # 3. BARRA DE HERRAMIENTAS (Botones)
    # ==========================================
    st.write("---") 
    
    # Definimos las columnas
    c1, col_camara, col_mic, c4 = st.columns([1, 2, 2, 1], gap="medium")
    
    # --- BOTÓN CÁMARA ---
    with col_camara:
        if st.button("📷 Cámara", use_container_width=True, key="btn_cam"):
            st.session_state.mostrar_camara = not st.session_state.mostrar_camara
    
    # --- BOTÓN MICRÓFONO ---
    with col_mic:
        # Capturamos el audio aquí. NO lo borraremos después.
        audio_data = mic_recorder(
            start_prompt="🎙️ Hablar",
            stop_prompt="⏹️ Enviar",
            just_once=True,
            key="grabadora"
        )
    
    # ==========================================
    # 4. ÁREA DE INPUTS (Cámara y Texto)
    # ==========================================
    
    # A. Visualizar Cámara (Si está activa)
    if st.session_state.mostrar_camara:
        st.write("### 📸 Tomar Foto")
        imagen_capturada = st.camera_input("Enfoca el medicamento", label_visibility="collapsed")
    
    # B. Caja de Texto (Siempre visible al final)
    texto_input = st.chat_input("Escribe aquí para consultar a Kivia...")
    
    
    # ==========================================
    # 5. CEREBRO: LÓGICA DE PROCESAMIENTO
    # ==========================================
    
    # --- CASO 1: VISIÓN (Prioridad Alta) ---
    if imagen_capturada:
        st.success("✅ Imagen recibida")
        # Evitamos procesar la misma foto dos veces
        if "ultima_foto_proc" not in st.session_state: st.session_state.ultima_foto_proc = None
        
        if imagen_capturada.getvalue() != st.session_state.ultima_foto_proc:
            prompt_usuario = "📸 "
            es_vision = True
            st.session_state.ultima_foto_proc = imagen_capturada.getvalue()
            
            with st.spinner("👁️ Kivia está analizando tu medicamento..."):
                # Llama a tu función de visión
                respuesta_ia = analizar_imagen(imagen_capturada.getvalue())
    
    # --- CASO 2: AUDIO (Si no hay foto nueva) ---
    elif audio_data and audio_data['id'] != st.session_state.ultimo_audio_id:
        # Verificamos que sea un audio nuevo
        st.session_state.ultimo_audio_id = audio_data['id']
        st.info("🎧 Procesando audio...")
        
        # Intentamos transcribir
        try:
            texto_transcrito = transcribir_audio(audio_data['bytes'])
            if texto_transcrito:
                prompt_usuario = texto_transcrito
                responder_con_voz = True # Si me hablas, te respondo con voz
                st.success(f"Te escuché: '{texto_transcrito}'")
        except Exception as e:
            st.error(f"Error en el audio: {e}")
    
    # --- CASO 3: TEXTO (Si escribiste en el chat) ---
    elif texto_input:
        prompt_usuario = texto_input
    
    
    # ==========================================
    # 6. GENERACIÓN DE RESPUESTA
    # ==========================================
    
    if prompt_usuario:
        # 1. Guardar mensaje del usuario
        st.session_state.chat_history.append(HumanMessage(content=prompt_usuario))
        
        # 2. Mostrar mensaje en pantalla (si es texto o audio)
        if not es_vision:
            icono = "🗣️" if responder_con_voz else "👤"
            st.chat_message("user").write(f"{icono} {prompt_usuario}")
    
        # 3. Si no tenemos respuesta aún (es texto/audio), consultamos al RAG
        if not es_vision and not respuesta_ia:
            with st.chat_message("assistant"):
                with st.spinner("Kivia está pensando..."):
                    
                    # Análisis de Riesgo
                    riesgo = analizar_riesgo(prompt_usuario)
                    
                    if "PELIGRO" in riesgo:
                        respuesta_ia = "🚨 EMERGENCIA: Tu seguridad es lo primero. Llama al 123."
                        st.error("⚠️ ALERTA DE SEGURIDAD DETECTADA")
                        responder_con_voz = True
                    elif "NEGATIVO" in riesgo:
                        respuesta_ia = responder_rag(f"[TRISTE] {prompt_usuario}", usuario_nombre)
                    else:
                        respuesta_ia = responder_rag(prompt_usuario, usuario_nombre)
    
        # 4. Mostrar respuesta final y reproducir audio si aplica
        if respuesta_ia:
            st.session_state.chat_history.append(AIMessage(content=respuesta_ia))
            
            if not es_vision:
                with st.chat_message("assistant"):
                    st.write(respuesta_ia)
                    if responder_con_voz:
                        audio_out = texto_a_voz(respuesta_ia)
                        if audio_out: 
                            st.audio(audio_out, format="audio/mp3", autoplay=True)
            
            # Recargar si fue visión para limpiar la cámara
            if es_vision: 
                st.rerun()
    
    # ==========================================
    # 7. MAPA (Al final)
    # ==========================================
    
    st.divider() # Una línea bonita para separar el chat del mapa
    social_view.mostrar_mapa_central()


# ==========================================
# BLOQUE C: CONTROLADOR PRINCIPAL
# ==========================================

def main():
    with st.sidebar:
        st.title("KIVIA v4")
        modo = st.radio("Menú", ["Planificador", "Chatbot"])
        st.divider()

    if modo == "Planificador":
        renderizar_planificador_interno()
        
    elif modo == "Chatbot":
        # Esta línea llama a tu código que envolviste arriba
        vista_chatbot()

if __name__ == "__main__":
    main()

















