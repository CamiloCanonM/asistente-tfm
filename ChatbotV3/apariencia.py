import streamlit as st
import os

def cargar_estilos_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"] { 
            font-family: 'Roboto', sans-serif; 
        }
        
        /* Fondo general claro */
        .stApp { 
            background-color: #F8F9FA; 
        }
        
        /* Sidebar blanco */
        [data-testid="stSidebar"] { 
            background-color: #ffffff; 
            border-right: 1px solid #e0e0e0; 
        }
        
        /* =============================================
           FIX PARA MÓVILES Y MODO OSCURO (NUEVO)
           ============================================= */
        /* Forzamos que los inputs (cajas de texto) sean blancos con letra negra */
        div[data-baseweb="input"] {
            background-color: #ffffff !important;
            border: 1px solid #ced4da !important;
            color: #000000 !important;
        }
        
        /* Color del texto dentro del input */
        input[type="text"], textarea {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important; /* Importante para Chrome Móvil */
            caret-color: #000000 !important; /* Color del cursor */
        }
        
        /* Color del texto de ayuda (Placeholder) para que se lea bien */
        ::placeholder {
            color: #666666 !important;
            opacity: 1 !important;
        }
        
        /* Arreglar etiqueta de los inputs (ej: "¿Qué buscas?") */
        label[data-testid="stLabel"], .stMarkdown p {
            color: #31333F !important; /* Gris oscuro casi negro */
        }

        /* =============================================
           BOTONES PERSONALIZADOS
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

        /* Ocultar menú default de Streamlit */
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
