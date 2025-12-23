¡Entendido! Lo que pasa es que la barra lateral (Sidebar) es como una "zona independiente" y a veces necesita reglas de estilo propias para imponerse al modo oscuro del móvil.

Vamos a aplicar la "Regla de Oro" para la barra lateral: Forzar que TODO lo que esté ahí dentro sea negro sobre fondo blanco, sin excepciones.

Actualiza tu archivo apariencia.py con este código final. He agregado una sección específica llamada /* FIX RADICAL PARA SIDEBAR */.

📄 Archivo: apariencia.py (Versión Final Móvil)
Copia y pega esto reemplazando todo lo anterior:

Python

import streamlit as st
import os

def cargar_estilos_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"] { 
            font-family: 'Roboto', sans-serif; 
        }
        
        /* 1. FONDO GENERAL */
        .stApp { 
            background-color: #F8F9FA; 
        }

        /* =============================================
           2. FIX RADICAL PARA SIDEBAR (BARRA IZQUIERDA)
           ============================================= */
        /* Forzar fondo blanco en la barra lateral */
        [data-testid="stSidebar"] { 
            background-color: #ffffff !important; 
            border-right: 1px solid #e0e0e0; 
        }

        /* FORZAR COLOR NEGRO en todos los textos de la barra lateral */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #262730 !important; /* Gris muy oscuro casi negro */
        }

        /* Inputs (Cajas de texto) dentro del Sidebar */
        [data-testid="stSidebar"] input {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }

        /* Fondo de las cajas de texto y selectores del Sidebar */
        [data-testid="stSidebar"] div[data-baseweb="input"],
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border-color: #cccccc !important;
            color: #000000 !important;
        }

        /* =============================================
           3. FIX GENERAL PARA MÓVILES (INPUTS)
           ============================================= */
        div[data-baseweb="input"] {
            background-color: #ffffff !important;
            border: 1px solid #ced4da !important;
            color: #000000 !important;
        }
        
        input[type="text"], textarea {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            caret-color: #000000 !important;
        }
        
        ::placeholder {
            color: #666666 !important;
            opacity: 1 !important;
        }
        
        /* Etiquetas generales */
        label[data-testid="stLabel"], .stMarkdown p {
            color: #31333F !important;
        }

        /* =============================================
           4. BOTONES PERSONALIZADOS
           ============================================= */
        div.stButton > button, [data-testid="stFileUploader"] button { 
            background: linear-gradient(90deg, #4A90E2 0%, #9013FE 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 15px !important;
            padding: 8px 20px !important;
            font-weight: 500 !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(144, 19, 254, 0.3) !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container { padding-top: 2rem; }
        
        </style>
    """, unsafe_allow_html=True)


def mostrar_header():
    ruta_logo = os.path.join(os.path.dirname(__file__), "logo.png")
    
    # CENTRADO DEL LOGO: Usamos 3 columnas para que no ocupe todo el ancho
    c1, c2, c3 = st.columns([1, 3, 3])
    
    with c2: # Ponemos la imagen solo en la columna del medio
        if os.path.exists(ruta_logo):
            st.image(ruta_logo, use_container_width=True)
        else:
            st.markdown("<h1 style='text-align: center; color:#4A90E2;'>🧬 KIVIA.AI</h1>", unsafe_allow_html=True)
