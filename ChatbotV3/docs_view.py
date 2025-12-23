def renderizar_documentos():
    st.subheader("📂 Biblioteca Inteligente")
    st.caption("Los archivos que subas aquí quedarán guardados para siempre.")

    # --- ZONA DE SUBIDA ---
    archivo = st.file_uploader("Subir nuevo documento", type=["pdf", "txt", "docx", "csv"])
    
    # === CORRECCIÓN DEL BUCLE INFINITO ===
    if archivo is not None:
        # Verificamos si este archivo específico YA lo acabamos de procesar
        if "ultimo_archivo_guardado" not in st.session_state:
            st.session_state.ultimo_archivo_guardado = ""
            
        # Solo guardamos si el nombre es diferente al último
        if st.session_state.ultimo_archivo_guardado != archivo.name:
            if guardar_archivo(archivo):
                st.session_state.ultimo_archivo_guardado = archivo.name # <--- MARCAMOS COMO LISTO
                st.success(f"✅ ¡{archivo.name} guardado con éxito!")
                st.rerun() # Ahora sí podemos recargar seguros
    # =====================================

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
