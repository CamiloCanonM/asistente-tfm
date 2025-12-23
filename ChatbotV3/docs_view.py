import streamlit as st
import os

# 1. CREAMOS LA CARPETA DONDE SE GUARDARÁN LOS ARCHIVOS
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
    return [f for f in os.listdir(CARPETA_DOCS) if os.path.isfile(os.path.join(CARPETA_DOCS, f))]

def eliminar_archivo(nombre_archivo):
    os.remove(os.path.join(CARPETA_DOCS, nombre_archivo))

def renderizar_documentos():
    st.subheader("📂 Mis Documentos")
    st.caption("Los archivos que subas aquí quedarán guardados en la nube.")

    # --- ZONA DE SUBIDA ---
    archivo = st.file_uploader("Subir nuevo documento", type=["pdf", "txt", "docx", "csv"])
    
    if archivo is not None:
        if guardar_archivo(archivo):
            st.success(f"✅ ¡{archivo.name} guardado con éxito!")
            st.rerun() # Recargamos para que aparezca en la lista

    st.divider()

    # --- ZONA DE LISTADO (MEMORIA) ---
    archivos_guardados = listar_archivos()
    
    if not archivos_guardados:
        st.info("📭 Aún no tienes documentos guardados.")
    else:
        st.write(f"**📚 Documentos disponibles ({len(archivos_guardados)}):**")
        
        for doc in archivos_guardados:
            col1, col2 = st.columns([0.8, 0.2])
            
            with col1:
                # Icono según extensión
                icono = "📄"
                if doc.endswith(".pdf"): icono = "📕"
                elif doc.endswith(".csv"): icono = "📊"
                elif doc.endswith(".txt"): icono = "📝"
                
                st.text(f"{icono} {doc}")
            
            with col2:
                if st.button("🗑️", key=f"del_{doc}", help="Eliminar archivo"):
                    eliminar_archivo(doc)
                    st.rerun()
