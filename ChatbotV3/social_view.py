import streamlit as st
import pandas as pd
import googlemaps
import time
from langchain_core.messages import AIMessage
from streamlit_js_eval import get_geolocation

# --- 1. CONEXIÓN ---
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# --- 2. BÚSQUEDA HÍBRIDA + TU POSICIÓN ---
def buscar_lugares_google(ciudad, palabra_clave, lat_gps=None, lon_gps=None):
    gmaps = obtener_cliente_google()
    if not gmaps: return None, "Error: Falta API Key."

    try:
        # A. DETERMINAR CENTRO
        if lat_gps is not None and lon_gps is not None:
            lat_centro, lon_centro = lat_gps, lon_gps
            origen_busqueda = "tu ubicación"
        else:
            geocode = gmaps.geocode(ciudad)
            if not geocode: return None, f"Ciudad no encontrada: {ciudad}"
            loc = geocode[0]['geometry']['location']
            lat_centro, lon_centro = loc['lat'], loc['lng']
            origen_busqueda = ciudad

        # B. BUSCAR LUGARES (Places API)
        res = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000,
            open_now=False
        )
        
        lugares = res.get('results', [])
        if not lugares: return None, f"No encontré {palabra_clave}."

        # C. CONSTRUIR DATOS (Mezclando Usuario y Lugares)
        data_completa = []

        # 1. INSERTAR TU UBICACIÓN (Punto AZUL) 🔵
        if lat_gps is not None and lon_gps is not None:
            data_completa.append({
                'latitude': float(lat_gps),
                'longitude': float(lon_gps),
                'name': "🔵 TU UBICACIÓN ACTUAL",
                'address': "Estás aquí ahora mismo",
                'rating': "📍",
                'place_id': None,
                'color': '#0000FF',  # <--- AZUL PURO
                'size': 50           # <--- MÁS GRANDE
            })

        # 2. INSERTAR LUGARES ENCONTRADOS (Puntos ROJOS) 🔴
        for l in lugares:
            data_completa.append({
                'latitude': float(l['geometry']['location']['lat']),
                'longitude': float(l['geometry']['location']['lng']),
                'name': l.get('name', 'Sin nombre'),
                'address': l.get('vicinity', 'Dirección desconocida'),
                'rating': l.get('rating', 'N/A'),
                'place_id': l.get('place_id'),
                'color': '#FF0000',  # <--- ROJO
                'size': 20           # <--- TAMAÑO NORMAL
            })
            
        return pd.DataFrame(data_completa), f"✅ Encontré {len(lugares)} {palabra_clave} cerca de {origen_busqueda}."

    except Exception as e:
        return None, f"Error: {str(e)}"

# --- 3. RENDERIZADO ---
def renderizar_sidebar():
    st.subheader("📍 Mapa Inteligente")
    
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None

    tipo = st.selectbox("Buscar:", ["Farmacias", "Hospitales", "Parques", "Gimnasios", "Restaurantes"])
    
    # CHECKBOX GPS
    usar_gps = st.checkbox("📍 Usar mi ubicación (GPS)")
    
    lat_usuario, lon_usuario = None, None
    ciudad = ""

    if usar_gps:
        loc = get_geolocation()
        if loc:
            lat_usuario = loc['coords']['latitude']
            lon_usuario = loc['coords']['longitude']
            st.caption(f"📡 GPS Activo: {lat_usuario:.4f}, {lon_usuario:.4f}")
        else:
            st.warning("⚠️ Esperando señal GPS...")
    else:
        ciudad = st.text_input("Ciudad:", "Bogota")

    # BOTÓN
    if st.button("🔍 LOCALIZAR", use_container_width=True):
        if usar_gps and lat_usuario is None:
            st.error("⚠️ Espera a que cargue el GPS o desactívalo.")
        else:
            with st.spinner("Triangulando..."):
                df, msg = buscar_lugares_google(ciudad, tipo, lat_gps=lat_usuario, lon_gps=lon_usuario)
                
                if df is not None:
                    st.session_state.mapa_data = df
                    st.session_state.chat_history.append(AIMessage(content=f"[MAPA]: {msg}"))
                    st.success(msg)
                else:
                    st.error(msg)
    
    # --- VISUALIZACIÓN ---
    if st.session_state.mapa_data is not None:
        df = st.session_state.mapa_data
        
        # 🗺️ EL MAPA CON COLORES (AZUL vs ROJO)
        # Usamos las columnas 'color' y 'size' que creamos arriba
        st.map(df, color='color', size='size') 

        st.divider()
        st.caption("Detalles:")

        for index, row in df.iterrows():
            # Filtramos para no mostrar tarjeta de "Tu Ubicación" (que no tiene place_id)
            if row['place_id'] is not None:
                with st.expander(f"📍 {row['name']} ({row['rating']}⭐)"):
                    st.write(f"🏠 {row['address']}")
                    link_google = f"https://www.google.com/maps/dir/?api=1&destination={row['name'].replace(' ', '+')}&destination_place_id={row['place_id']}"
                    st.markdown(f"[🚗 **CÓMO LLEGAR**]({link_google})")
        
        if st.button("Limpiar"):
            st.session_state.mapa_data = None
            st.rerun()
