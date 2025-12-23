import streamlit as st
import os

def cargar_estilos_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        /* =============================================
           1. REGLA SUPREMA DE TEXTO (SOLUCIÓN AL TEXTO INVISIBLE)
           ============================================= */
        html, body, [class*="css"], .stMarkdown, p, h1, h2, h3,span, div { 
            font-family: 'Roboto', sans-serif; 
            color: #000000 !important; /* Fuerza bruta: TODO TEXTO NEGRO */
        }
        
        .stApp { 
            background-color: #F8F9FA !important; 
        }

        /* =============================================
           2. ARREGLO DE CHAT (Burbujas y Texto)
           ============================================= */
        /* Mensajes del Usuario y del Asistente */
        [data-testid="stChatMessage"] {
            background-color: #ffffff !important;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
        }
        
        /* El texto dentro de los mensajes de chat */
        [data-testid="stChatMessageContent"] p, 
        [data-testid="stChatMessageContent"] div {
            color: #000000 !important;
        }
        
        /* Iconos del chat (Avatar) */
        [data-testid="stChatMessageAvatar"] {
            background-color: #ffffff !important;
            border: 1px solid #ccc;
        }

        /* =============================================
           3. ARREGLO DE "BROWSE FILES" Y UPLOADER
           ============================================= */
        [data-testid="stFileUploader"] {
            padding: 10px;
            background-color: #ffffff !important;
            border-radius: 10px;
        }
        
        [data-testid="stFileUploader"] section {
            background-color: #f0f2f6 !important; /* Fondo gris clarito para la zona de soltar */
        }
        
        /* Texto "Drag and drop..." y "Limit 200MB" */
        [data-testid="stFileUploader"] div, 
        [data-testid="stFileUploader"] span, 
        [data-testid="stFileUploader"] small {
            color: #31333F !important; /* Gris oscuro */
        }

        /* El botón "Browse files" específico */
        [data-testid="stFileUploader"] button {
            color: #ffffff !important; /* Texto del botón BLANCO */
        }

        /* =============================================
           4. CAJAS DE TEXTO (INPUTS) Y CHAT INPUT
           ============================================= */
        input[type="text"], textarea, [data-baseweb="input"], [data-baseweb="base-input"] {
            background-color: #ffffff !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            caret-color: #000000 !important;
            border: 1px solid #cccccc !important;
        }
        
        [data-testid="stChatInput"] {
            background-color: #ffffff !important;
        }

        /* =============================================
           5. BOTONES (CAMARA, HABLAR, ETC)
           ============================================= */
        /* Estilo general para botones */
        div.stButton > button { 
            background: linear-gradient(90deg, #4A90E2 0%, #9013FE 100%) !important;
            color: #ffffff !important; /* Texto blanco forzado */
            border: none !important;
            border-radius: 15px !important;
            padding: 8px 20px !important;
            font-weight: bold !important;
            text-shadow: none !important;
        }

        /* Arreglo específico para el botón de audio (mic_recorder a veces usa clases distintas) */
        button {
            color: #000000; /* Por defecto negro para botones raros, salvo los stButton */
        }

        /* =============================================
           6. SIDEBAR (BARRA LATERAL)
           ============================================= */
        [data-testid="stSidebar"] { 
            background-color: #ffffff !important; 
        }
        [data-testid="stSidebar"] * {
            color: #262730 !important; /* Todo texto en sidebar: Gris oscuro */
        }
        
        /* Ocultar elementos extra */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
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
