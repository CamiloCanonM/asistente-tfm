import streamlit as st
import os

# --- TUS FUNCIONES EXISTENTES (Estilos y Header) ---
def cargar_estilos_css():
    st.markdown("""
        <style>
        /* ... (Aquí va todo el CSS que pegaste arriba) ... */
        .stApp { background-color: #F8F9FA; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }

        /* Estilo unificado de botones */
        .stButton > button, 
        [data-testid="stFileUploader"] button,
        [data-testid="stLinkButton"] a { 
            background: linear-gradient(45deg, #4A90E2, #9013FE);
            color: white !important;
            border: none;
            border-radius: 20px;
            padding: 8px 20px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-weight: 500;
            text-decoration: none !important;
            display: inline-flex;
            justify-content: center;
            align-items: center;
        }

        .stButton > button:hover, 
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stLinkButton"] a:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            color: white !important;
        }
        
        /* Ocultar elementos extra */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container { padding-top: 1rem; }
        </style>
    """, unsafe_allow_html=True)

def mostrar_header():
    """Muestra el logo si existe, o un título si no"""
    ruta_logo = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(ruta_logo):
        st.image(ruta_logo, use_container_width=True)
    else:
        # Fallback elegante si no hay imagen
        st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🧬 KIVIA.AI</h1>", unsafe_allow_html=True)

# --- NUEVA FUNCIÓN: Área de Acciones (Cámara y Hablar) ---

def mostrar_area_acciones():
    """
    Muestra los botones de Cámara y Hablar centrados y organiza
    la lógica de visualización (qué pasa cuando haces clic).
    """
    st.write("---") # Separador visual sutil
    
    # 1. Definir columnas para centrar (Estructura 1 - 2 - 2 - 1)
    # Los 'gap' ayudan a que no se vean pegados
    c1, c_camara, c_micro, c4 = st.columns([1, 2, 2, 1], gap="medium")

    # 2. Inicializar estado para la cámara (toggle)
    if "camara_activa" not in st.session_state:
        st.session_state.camara_activa = False

    # 3. Botón CÁMARA
    with c_camara:
        # use_container_width=True hace que el botón llene la columna
        # y luzca como una "tarjeta" con tu estilo degradado.
        if st.button("📷 Activar Cámara", use_container_width=True):
            st.session_state.camara_activa = not st.session_state.camara_activa

    # 4. Botón HABLAR
    with c_micro:
        if st.button("🎙️ Hablar", use_container_width=True):
            st.info("Escuchando... (Simulación)")

    # 5. Área de despliegue de la cámara (se muestra debajo si está activa)
    if st.session_state.camara_activa:
        st.write("") # Espacio
        with st.container():
            st.markdown("##### 📸 Vista de Cámara")
            imagen_capturada = st.camera_input("Toma una foto", label_visibility="collapsed")
            
            if imagen_capturada:
                st.success("Imagen capturada correctamente")
                # Aquí iría tu lógica de procesamiento

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Kivia.AI")
    
    cargar_estilos_css()  # 1. Aplicamos tu diseño
    mostrar_header()      # 2. Mostramos el logo
    
    st.write("### Hola, Usuario 👋")
    
    # Ejemplo de tarjeta de mapa (como en tu imagen)
    with st.expander("🗺️ Estado del Mapa", expanded=True):
        st.success("Mapa de **droguerías** cargado correctamente.")

    mostrar_area_acciones() # 3. Mostramos los botones centrados

    # Chat input al final
    st.chat_input("Escribe aquí para consultar a Kivia...")
