import streamlit as st
import os

# 1. CREAMOS LA CARPETA
CARPETA_DOCS = "base_conocimiento"
if not os.path.exists(CARPETA_DOCS):
    os.makedirs(CARPETA_DOCS)

def guardar_archivo(archivo_subido):
    try:
        ruta_completa = os.path.join(CARPETA_DOCS, archivo_subido.name)
        with open(ruta_completa, "wb") as f:
            f.write(archivo_subido.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error guardando archivo: {e}")
        return False

def listar_archivos():
    # Lista solo archivos reales, no carpetas
    return [f for f in os.listdir(CARPETA_DOCS) if os.path.isfile(os.path.join(CARPETA_DOCS, f))]

def eliminar_archivo(nombre_archivo):
    try:
        os.remove(os.path.join(CARPETA_DOCS, nombre_archivo))
        st.toast(f"🗑️ Archivo eliminado: {nombre_archivo}")
    except Exception as e:
        st.error(f"Error eliminando: {e}")

def renderizar_documentos():
    st.subheader("📂 Biblioteca Inteligente")
    st.caption("Sube tus PDFs o archivos aquí.")

    # --- ZONA DE SUBIDA (CON FIX DE BUCLE INFINITO) ---
    archivo = st.file_uploader("Subir documento", type=["pdf", "txt", "docx", "csv", "xlsx"])
    
    if archivo is not None:
        # Inicializar estado si no existe
        if "ultimo_archivo_guardado" not in st.session_state:
            st.session_state.ultimo_archivo_guardado = ""
            
        # Solo guardamos si es un archivo DIFERENTE al último procesado
        if st.session_state.ultimo_archivo_guardado != archivo.name:
            if guardar_archivo(archivo):
                st.session_state.ultimo_archivo_guardado = archivo.name
                st.success(f"✅ Guardado: {archivo.name}")
                st.rerun() # Recarga segura

    st.divider()

    # --- LISTADO DE ARCHIVOS ---
    archivos = listar_archivos()
    
    if not archivos:
        st.info("📭 Carpeta vacía.")
    else:
        st.write(f"**📚 Archivos ({len(archivos)}):**")
        for doc in archivos:
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                st.text(f"📄 {doc}")
            with c2:
                if st.button("X", key=f"del_{doc}", help="Borrar"):
                    eliminar_archivo(doc)
                    st.rerun()
