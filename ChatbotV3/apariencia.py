import streamlit as st
import os

def cargar_estilos_css():
    st.markdown("""
        <style>
        /* 1. Fondo y estructura general */
        .stApp { background-color: #F8F9FA; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }

        /* 2. ESTILO BOTONES (Incluido Browse Files) */
        .stButton > button, 
        [data-testid="stFileUploader"] button {
            background: linear-gradient(45deg, #4A90E2, #9013FE);
            color: white !important;
            border: none;
            border-radius: 20px;
            padding: 8px 20px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-weight: 500;
        }

        /* 3. Hover */
        .stButton > button:hover, 
        [data-testid="stFileUploader"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            color: white !important;
        }

        /* 4. Inputs */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #E0E0E0;
            padding: 10px;
        }
        
        /* 5. Carga de archivo */
        [data-testid="stFileUploader"] section {
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 15px;
            border: 1px dashed #4A90E2;
        }

        /* 6. Ocultar menú default */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 7. Tarjetas */
        div[data-testid="stExpander"] {
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-radius: 10px;
            background-color: white;
        }
        
        /* 8. Quitar padding superior excesivo */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        </style>
    """, unsafe_allow_html=True)

def mostrar_header():
    """Función para cargar y mostrar el logo/banner"""
    ruta_logo = os.path.join(os.path.dirname(__file__), "logo.png")
    
    if os.path.exists(ruta_logo):
        st.image(ruta_logo, use_container_width=True)
    else:
        # Un fallback por si no encuentra la imagen
        st.warning("⚠️ No encuentro el archivo logo.png")
