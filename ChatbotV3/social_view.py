import streamlit as st
import pandas as pd
import googlemaps
import time
from langchain_core.messages import AIMessage

# --- 1. CONEXIÓN ---
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# --- 2. BÚSQUEDA ENRIQUECIDA ---
def buscar_lugares_google(ciudad, palabra_clave):
    gmaps = obtener_cliente_google()
    
    if not gmaps: return None, "Error: Falta API Key."

    try:
        # A. Geocodificar ciudad
        geocode = gmaps.geocode(ciudad)
        if not geocode: return None, f"Ciudad no encontrada: {ciudad}"
        
        loc = geocode[0]['geometry']['location']
        lat_centro, lon_centro = loc['lat'], loc['lng']

        # B. Buscar Lugares (Solicitamos más detalles)
        res = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000,
            open_now=False # Cambiar a True si solo quieres abiertos
        )
        
        lugares = res.get('results', [])
        if not lugares: return None, f"No encontré {palabra_clave}."

        # C. Extraer DATOS COMPLETOS (Nombre, Dirección, Rating, ID)
        data_completa = []
        for l in lugares:
            lat = l['geometry']['location']['lat']
            lng = l['geometry']['location']['lng']
            nombre = l.get('name', 'Sin nombre')
            direccion = l.get('vicinity', 'Dirección desconocida')
            rating = l.get('rating', 'N/A')
            place_id = l.get('place_id') # Clave para el enlace de "Cómo llegar"
            
            data_completa.append({
                'latitude': float(lat),
                'longitude': float(lng),
                'name': nombre,
                'address': direccion,
                'rating': rating,
                'place_id': place_id
            })
            
        return pd.DataFrame(data_completa), f"✅ Encontré {len(data_completa)} {palabra_clave}."

    except Exception as e:
        return None, f"Error: {str(e)}"

# --- 3. RENDERIZADO INTERACTIVO ---
def renderizar_sidebar():
    st.subheader("📍 Mapa Inteligente")
    
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None

    ciudad = st.text_input("Ciudad:", "Bogota")
    tipo = st.selectbox("Buscar:", ["Farmacias", "Hospitales", "Parques", "Gimnasios", "Restaurantes"])
    
    # BOTÓN
    if st.button("🔍 BUSCAR Y OBTENER RUTA", use_container_width=True):
        with st.spinner("Localizando sitios..."):
            df, msg = buscar_lugares_google(ciudad, tipo)
            if df is not None:
                st.session_state.mapa_data = df
                st.session_state.chat_history.append(AIMessage(content=f"[MAPA]: {msg} (Ver detalles en panel lateral)"))
                st.success(msg)
            else:
                st.error(msg)
    
    # --- VISUALIZACIÓN DE RESULTADOS ---
    if st.session_state.mapa_data is not None:
        df = st.session_state.mapa_data
        
        # 1. El Mapa Visual (Puntos)
        st.map(df, size=20, color='#0044ff') 

        st.divider()
        st.caption(f"Resultados detallados ({len(df)}):")

        # 2. LISTA INTERACTIVA (Aquí está la magia)
        # Mostramos los primeros 5 para no saturar, o todos con scroll
        for index, row in df.iterrows():
            with st.expander(f"📍 {row['name']} ({row['rating']}⭐)"):
                st.write(f"🏠 **Dirección:** {row['address']}")
                
                # CREAR ENLACE "CÓMO LLEGAR" DE GOOGLE
                # Este enlace abre la app de Google Maps directamente con la ruta
                link_google = f"https://www.google.com/maps/dir/?api=1&destination={row['name'].replace(' ', '+')}&destination_place_id={row['place_id']}"
                
                st.markdown(f"[🚗 **Ir ahora (Google Maps)**]({link_google})")
        
        if st.button("Limpiar Resultados"):
            st.session_state.mapa_data = None
            st.rerun()
