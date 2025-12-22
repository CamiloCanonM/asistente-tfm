import streamlit as st
import pandas as pd
import googlemaps
import time
from langchain_core.messages import AIMessage
from streamlit_js_eval import get_geolocation # <--- NUEVA IMPORTACIÓN

# --- 1. CONEXIÓN ---
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# --- 2. BÚSQUEDA HÍBRIDA (TEXTO O GPS) ---
def buscar_lugares_google(ciudad, palabra_clave, lat_gps=None, lon_gps=None):
    gmaps = obtener_cliente_google()
    if not gmaps: return None, "Error: Falta API Key."

    try:
        # A. DETERMINAR EL CENTRO DE BÚSQUEDA
        if lat_gps is not None and lon_gps is not None:
            # Opción 1: Usamos GPS directo
            lat_centro, lon_centro = lat_gps, lon_gps
            origen_busqueda = "tu ubicación actual"
        else:
            # Opción 2: Buscamos la ciudad escrita
            geocode = gmaps.geocode(ciudad)
            if not geocode: return None, f"Ciudad no encontrada: {ciudad}"
            loc = geocode[0]['geometry']['location']
            lat_centro, lon_centro = loc['lat'], loc['lng']
            origen_busqueda = ciudad

        # B. BUSCAR LUGARES (Places API)
        res = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000, # 2km a la redonda
            open_now=False
        )
        
        lugares = res.get('results', [])
        if not lugares: return None, f"No encontré {palabra_clave} cerca de {origen_busqueda}."

        # C. EXTRAER DATOS
        data_completa = []
        for l in lugares:
            data_completa.append({
                'latitude': float(l['geometry']['location']['lat']),
                'longitude': float(l['geometry']['location']['lng']),
                'name': l.get('name', 'Sin nombre'),
                'address': l.get('vicinity', 'Dirección desconocida'),
                'rating': l.get('rating', 'N/A'),
                'place_id': l.get('place_id')
            })
            
        return pd.DataFrame(data_completa), f"✅ Encontré {len(data_completa)} {palabra_clave} cerca de {origen_busqueda}."

    except Exception as e:
        return None, f"Error: {str(e)}"

# --- 3. RENDERIZADO CON GPS ---
def renderizar_sidebar():
    st.subheader("📍 Mapa Inteligente")
    
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None

    # --- ZONA DE CONFIGURACIÓN ---
    tipo = st.selectbox("Buscar:", ["Farmacias", "Hospitales", "Parques", "Gimnasios", "Restaurantes"])
    
    # CHECKBOX GPS
    usar_gps = st.checkbox("📍 Usar mi ubicación actual (GPS)")
    
    lat_usuario, lon_usuario = None, None
    ciudad = ""

    if usar_gps:
        # Esto pide permiso al navegador
        loc = get_geolocation()
        if loc:
            lat_usuario = loc['coords']['latitude']
            lon_usuario = loc['coords']['longitude']
            st.caption(f"📡 GPS Detectado: {lat_usuario:.4f}, {lon_usuario:.4f}")
        else:
            st.warning("⚠️ Esperando permiso de ubicación...")
    else:
        ciudad = st.text_input("Ciudad:", "Bogota")

    # --- BOTÓN DE BÚSQUEDA ---
    if st.button("🔍 BUSCAR AHORA", use_container_width=True):
        
        # Validar que tengamos datos para buscar
        if usar_gps and lat_usuario is None:
            st.error("⚠️ Aún no tengo tu ubicación. Espera un segundo o desactiva el GPS.")
        else:
            with st.spinner("Escaneando zona..."):
                # Llamamos a la función con o sin coordenadas
                df, msg = buscar_lugares_google(ciudad, tipo, lat_gps=lat_usuario, lon_gps=lon_usuario)
                
                if df is not None:
                    st.session_state.mapa_data = df
                    st.session_state.chat_history.append(AIMessage(content=f"[MAPA]: {msg}"))
                    st.success(msg)
                else:
                    st.error(msg)
    
    # --- RESULTADOS ---
    if st.session_state.mapa_data is not None:
        df = st.session_state.mapa_data
        st.map(df, size=20, color='#0044ff') 

        st.divider()
        st.caption(f"Resultados más cercanos:")

        for index, row in df.iterrows():
            with st.expander(f"📍 {row['name']} ({row['rating']}⭐)"):
                st.write(f"🏠 {row['address']}")
                # Enlace corregido para abrir la ruta
                link_google = f"https://www.google.com/maps/dir/?api=1&destination={row['latitude']},{row['longitude']}&destination_place_id={row['place_id']}"
                st.markdown(f"[🚗 **LLÉVAME ALLÍ**]({link_google})")
        
        if st.button("Limpiar"):
            st.session_state.mapa_data = None
            st.rerun()
