
import streamlit as st
import kivia_backend as backend # Importamos el cerebro que acabamos de crear

def renderizar_planificador():
    st.title("🧠 Diagnóstico Profundo Kivia")
    st.markdown("Responde este cuestionario para que la IA analice tus patrones ocultos.")

    # --- INICIO DEL FORMULARIO ---
    with st.form("cuestionario_kivia"):
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("⚡ Estado Físico")
            energia_in = st.select_slider("Nivel de Energía diario", 
                                        options=["Muy bajo", "Bajo", "Moderado", "Alto", "Muy Alto"], value="Moderado")
            ejercicio_in = st.select_slider("Frecuencia de Ejercicio", 
                                          options=["Nunca", "1 día/sem", "3 días/sem", "5+ días/sem"], value="1 día/sem")
            sueño_in = st.slider("Calidad de Sueño (0-100)", 0, 100, 70)

        with c2:
            st.subheader("🧘 Estado Mental")
            estres_in = st.select_slider("Nivel de Estrés", 
                                       options=["Zen", "Bajo", "Manejable", "Alto", "Crítico"], value="Manejable")
            animo_in = st.slider("Estado de Ánimo General (0-100)", 0, 100, 60)
            disciplina_in = st.select_slider("Autodisciplina percibida", 
                                           options=["Baja", "Variable", "Alta", "Hierro"], value="Variable")

        # Botón de envío (Dentro del form para no recargar a cada click)
        submitted = st.form_submit_button("🚀 Analizar mis Probabilidades", type="primary")

    # --- PROCESAMIENTO AL ENVIAR ---
    if submitted:
        modelo = backend.cargar_cerebro_completo()
        
        if modelo:
            try:
                # 1. Mapeo de Texto a Números
                map_energia = {"Muy bajo": 0.1, "Bajo": 0.3, "Moderado": 0.5, "Alto": 0.8, "Muy Alto": 1.0}
                map_ejer = {"Nunca": 0.0, "1 día/sem": 0.3, "3 días/sem": 0.7, "5+ días/sem": 1.0}
                map_estres = {"Zen": 0.0, "Bajo": 0.2, "Manejable": 0.5, "Alto": 0.8, "Crítico": 1.0}
                map_disci = {"Baja": 0.2, "Variable": 0.5, "Alta": 0.8, "Hierro": 1.0}

                respuestas_dict = {
                    "energia": map_energia[energia_in],
                    "ejercicio": map_ejer[ejercicio_in],
                    "sueño": sueño_in / 100.0,
                    "estres": map_estres[estres_in],
                    "animo": animo_in / 100.0,
                    "disciplina": map_disci[disciplina_in]
                }

                # 2. Llamada al Backend para generar el vector de 50
                vector_50 = backend.procesar_cuestionario_inteligente(respuestas_dict)

                # 3. Predicciones
                datos_escalados = modelo.scaler.transform(vector_50)
                datos_pca = modelo.pca.transform(datos_escalados)
                
                prob_exito = modelo.xgb_model.predict_proba(datos_pca)[0, 1]
                score_raw = modelo.regression_model.predict(datos_pca)[0]
                kivia_score = int(max(0, min(100, score_raw)))

                # 4. Guardar en Sesión y Mostrar
                st.session_state['kivia_data'] = {
                    "score": kivia_score, 
                    "prob": round(prob_exito, 2),
                    "perfil": respuestas_dict
                }
                
                st.divider()
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.metric("Kivia Score", f"{kivia_score}/100")
                    st.progress(kivia_score / 100)
                
                with col_res2:
                    if prob_exito > 0.7:
                        st.success(f"🌟 **Alta Probabilidad ({prob_exito:.0%})**: Tienes el perfil ideal.")
                    elif prob_exito > 0.4:
                        st.warning(f"⚖️ **Probabilidad Media ({prob_exito:.0%})**: Se requieren ajustes.")
                    else:
                        st.error(f"🛡️ **Probabilidad Baja ({prob_exito:.0%})**: Empieza despacio.")

            except Exception as e:
                st.error(f"Error en el análisis: {e}")
        else:
            st.error("❌ No se encontró el modelo 'habit_model.pkl'.")
