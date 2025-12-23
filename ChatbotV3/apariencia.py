import streamlit as st
import os

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="KIVIA.AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ESTILOS CSS (DISEÑO)
def cargar_estilos():
    st.markdown("""
        <style>
        /* Importar fuente moderna */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }

        /* --- ESTILO DE BOTONES (Gradiente Morado/Azul) --- */
        /* Aplica a todos los botones: Buscar, Sincronizar, Cámara, Hablar */
        div.stButton > button {
            background: linear-gradient(90deg, #4A90E2 0%, #9013FE 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease !important;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(144, 19, 254, 0.3) !important;
            background: linear-gradient(90deg, #357ABD 0%, #7B0FCC 100%) !important;
        }

        /* Estilo específico para el botón de carga de archivos */
        [data-testid="stFileUploader"] button {
            background: linear-gradient(90deg, #4A90E2, #9013FE);
            color: white;
            border: none;
        }

        /* --- CONTENEDORES Y TEXTOS --- */
        .stTextInput > div > div > input {
            border-radius: 10px;
        }

        /* Mensajes de éxito (El mapa cargado) */
        .stSuccess {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 10px;
        }

        /* Ocultar elementos por defecto de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        </style>
    """, unsafe_allow_html=True)

# 3. BARRA LATERAL (SIDEBAR)
def crear_sidebar():
    with st.sidebar:
        st.header("🌍 Módulo de Georreferenciación")
        
        st.text_input("¿Qué buscas?", placeholder="Ej: Pizza, Farmacia, Cine...")
        
        # Checkbox personalizado alineado
        col_check, col_lbl = st.columns([0.15, 0.85])
        with col_check:
            st.checkbox("", value=False, key="gps_check")
        with col_lbl:
            st.write("Usar GPS")
            
        st.text_input("Ciudad:", value="Bogota")
        
        st.write("") # Espacio
        st.button("🔍 BUSCAR", use_container_width=True)
        
        st.markdown("---")
        
        st.header("📂 Mis Documentos")
        st.write("Sube tus PDFs o archivos aquí.")
        st.file_uploader("Subir documento", label_visibility="collapsed")
        
        # Simulación de archivo cargado
        with st.expander("Archivos (1)", expanded=True):
            st.write("📄 medicamentos.pdf")
            
        st.markdown("---")
        
        st.header("⌚ Monitor Wearable")
        st.button("🔄 Sincronizar Reloj", use_container_width=True)

# 4. ÁREA PRINCIPAL
def crear_main_area():
    # Banner (Simulado con HTML para no depender de una imagen externa, 
    # si tienes la imagen usa st.image('logo.png'))
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="background: linear-gradient(90deg, #A1C4FD 0%, #C2E9FB 100%); 
                        padding: 40px; border-radius: 15px; color: #333;">
                <h1 style="margin:0; color: #2c3e50;">🧬 KIVIA.AI</h1>
                <p style="margin:0;">Asistente Conversacional</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.write("### Hola, Usuario 👋")

    # Mensaje de estado del mapa
    st.success("✅ Mapa de **droguerías** cargado abajo.")

    st.write("---")

    # --- AQUÍ ESTÁ LA CORRECCIÓN DE LOS BOTONES ---
    # Usamos columnas para centrarlos y use_container_width para que sean anchos
    c1, c_cam, c_hablar, c4 = st.columns([1, 2, 2, 1], gap="medium")

    with c_cam:
        # Botón CÁMARA (Ya NO es un selectbox)
        if st.button("📷 Cámara", use_container_width=True):
            st.session_state.accion = "camara"

    with c_hablar:
        # Botón HABLAR
        if st.button("🎙️ Hablar", use_container_width=True):
            st.session_state.accion = "hablar"

    # Lógica de visualización según el botón presionado
    if "accion" in st.session_state:
        if st.session_state.accion == "camara":
            st.info("Abriendo módulo de cámara...")
            st.camera_input("Tomar foto", label_visibility="hidden")
            
        elif st.session_state.accion == "hablar":
            st.info("🎙️ Escuchando... Habla ahora.")

    # Chat Input al final
    st.chat_input("Escribe aquí...")

# 5. EJECUCIÓN
if __name__ == "__main__":
    cargar_estilos()
    crear_sidebar()
    crear_main_area()
