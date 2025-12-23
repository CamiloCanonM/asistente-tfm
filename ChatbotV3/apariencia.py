import streamlit as st
import os

def cargar_estilos_css():
    st.markdown("""
        <style>
        /* 1. Fuente y Estructura Base */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        
        /* Color de fondo general */
        .stApp { background-color: #F8F9FA; }
        
        /* 2. BARRA LATERAL (Sidebar) - Fondo blanco y borde */
        [data-testid="stSidebar"] { 
            background-color: #ffffff; 
            border-right: 1px solid #e0e0e0; 
        }

        /* 3. BOTONES CON DEGRADADO (ESTILO KIVIA) */
        /* Esto afecta a: Botones normales, Subir archivo y Enlaces */
        div.stButton > button, 
        [data-testid="stFileUploader"] button,
        [data-testid="stLinkButton"] a { 
            background: linear-gradient(90deg, #4A90E2 0%, #9013FE 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 15px !important;
            padding: 8px 20px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            font-weight: 500 !important;
            text-decoration: none !important;
        }

        /* Efecto al pasar el mouse (Hover) */
        div.stButton > button:hover, 
        [data-testid="stFileUploader"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(144, 19, 254, 0.3) !important;
            background: linear-gradient(90deg, #357ABD 0%, #7B0FCC 100%) !important;
            color: white !important;
        }

        /* 4. Inputs (Cajas de texto) */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #E0E0E0;
            padding: 10px;
        }
        
        /* 5. Mensajes de Éxito (Caja verde del mapa) */
        .stSuccess {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 10px;
        }

        /* 6. Ocultar elementos por defecto de Streamlit (Menu y Footer) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Ajustar margen superior */
        .block-container { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)

def mostrar_header():
    """Muestra el logo centrado y con tamaño controlado"""
    ruta_logo = os.path.join(os.path.dirname(__file__), "logo.png")
    
    # Usamos 3 columnas: [Espacio, LOGO, Espacio]
    # El [1, 1, 1] significa que el logo ocupará el 50% del ancho central
    c1, c2, c3 = st.columns([1, 2, 1]) 
    
    with c2:
        if os.path.exists(ruta_logo):
            st.image(ruta_logo, use_container_width=True)
        else:
            st.markdown("<h1 style='text-align: center; color:#4A90E2;'>🧬 KIVIA.AI</h1>", unsafe_allow_html=True)
