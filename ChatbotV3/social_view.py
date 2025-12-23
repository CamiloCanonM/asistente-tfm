import streamlit as st
import pandas as pd
import googlemaps
from streamlit_js_eval import get_geolocation
from langchain_core.messages import AIMessage

# --- 1. CONEXIÓN ---
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# --- 2. BÚSQUEDA ---
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

        # B. BUSCAR LUGARES
        res = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000,
            open_now=False
        )
        
        lugares = res.get('results', [])
        if not lugares: return None, f"No encontré {palabra_clave}."

        # C. CONSTRUIR DATOS
        data_completa = []

        # 1. TU UBICACIÓN (AZUL) 🔵
        if lat_gps is not None and lon_gps is not None:
            data_completa.append({
                'latitude': float(lat_gps),
                'longitude': float(lon_gps),
                'name': "🔵 TU UBICACIÓN",
                'address': "Aquí estás tú",
                'rating': "YO",
                'place_id': "USER_LOC", # ID especial para ti
                'color': '#0000FF',     # Azul
                'size': 40              # Grande
            })

        # 2. LUGARES (ROJO POR DEFECTO) 🔴
        for l in lugares:
            data_completa.append({
                'latitude': float(l['geometry']['location']['lat']),
                'longitude': float(l['geometry']['location']['lng']),
                'name': l.get('name', 'Sin nombre'),
                'address': l.get('vicinity', 'Dirección desconocida'),
                'rating': l.get('rating', 'N/A'),
                'place_id': l.get('place_id'),
                'color': '#FF0000',     # Rojo
                'size': 20              # Normal
            })
            
        return pd.DataFrame(data_completa), f"✅ Encontré {len(lugares)} sitios."

    except Exception as e:
        return None, f"Error: {str(e)}"

# --- 3. RENDERIZADO ---
def renderizar_sidebar():
    st.subheader("📍 Mapa Interactivo")
    
    # Memoria de datos y de SELECCIÓN
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None
    if "lugar_seleccionado_id" not in st.session_state:
        st.session_state.lugar_seleccionado_id = None

    tipo = st.selectbox("Buscar:", ["Farmacias", "Hospitales", "Parques", "Gimnasios", "Restaurantes"])
    usar_gps = st.checkbox("📍 Usar mi ubicación (GPS)")
    
    lat_usuario, lon_usuario = None, None
    ciudad = ""

    if usar_gps:
        loc = get_geolocation()
        if loc:
            lat_usuario = loc['coords']['latitude']
            lon_usuario = loc['coords']['longitude']
        else:
            st.warning("⚠️ Activando GPS...")
    else:
        ciudad = st.text_input("Ciudad:", "Bogota")

    # BOTÓN BUSCAR
    if st.button("🔍 BUSCAR", use_container_width=True):
        if usar_gps and lat_usuario is None:
            st.error("Espera a tener señal GPS.")
        else:
            with st.spinner("Buscando..."):
                df, msg = buscar_lugares_google(ciudad, tipo, lat_gps=lat_usuario, lon_gps=lon_usuario)
                if df is not None:
                    st.session_state.mapa_data = df
                    st.session_state.lugar_seleccionado_id = None # Resetear selección al buscar de nuevo
                    st.session_state.chat_history.append(AIMessage(content=f"[MAPA]: {msg}"))
                    st.rerun() # Recargamos para mostrar el mapa limpio

    # --- LÓGICA DE VISUALIZACIÓN ---
    if st.session_state.mapa_data is not None:
        
        # 1. Hacemos una COPIA del dataframe para modificar colores sin dañar el original
        df_display = st.session_state.mapa_data.copy()

        # 2. APLICAR EL "FOCO" (Highlight)
        # Si el usuario seleccionó un lugar específico, lo pintamos VERDE y GIGANTE
        selected_id = st.session_state.lugar_seleccionado_id
        
        if selected_id:
            # Buscamos la fila que coincide con el ID seleccionado y le cambiamos color/tamaño
            mask = df_display['place_id'] == selected_id
            df_display.loc[mask, 'color'] = '#00FF00' # 🟢 VERDE NEÓN
            df_display.loc[mask, 'size'] = 60         # 🟢 GIGANTE
            
            st.info("🎯 Mostrando ubicación seleccionada en VERDE.")

        # 3. PINTAR EL MAPA
        st.map(df_display, color='color', size='size') 
        
        # Botón para quitar el foco y ver todo normal
        if selected_id and st.button("🔙 Ver todos normal"):
            st.session_state.lugar_seleccionado_id = None
            st.rerun()

        st.divider()
        st.caption("Lista de resultados:")

        # 4. LISTA DE TARJETAS
        # Filtramos para no mostrar la tarjeta de "Tu ubicación" en la lista
        df_lugares = st.session_state.mapa_data[st.session_state.mapa_data['place_id'] != "USER_LOC"]

        for index, row in df_lugares.iterrows():
            with st.expander(f"📍 {row['name']} ({row['rating']}⭐)"):
                st.write(f"🏠 {row['address']}")
                
                col_a, col_b = st.columns(2)
                
                # BOTÓN A: UBICAR EN EL MAPA ARRIBA
                with col_a:
                    if st.button("🎯 Ver en Mapa", key=f"btn_loc_{index}"):
                        st.session_state.lugar_seleccionado_id = row['place_id']
                        st.rerun() # Recargar para pintar el punto verde
                
                # BOTÓN B: IR CON GOOGLE MAPS
                with col_b:
                    link_google = f"https://www.google.com/maps/dir/?api=1&destination={row['name'].replace(' ', '+')}&destination_place_id={row['place_id']}"
                    st.markdown(f"[🚗 **Ir Ahora**]({link_google})")
        
        if st.button("🗑️ Limpiar Todo"):
            st.session_state.mapa_data = None
            st.session_state.lugar_seleccionado_id = None
            st.rerun()
