import streamlit as st
import pandas as pd
import googlemaps
from streamlit_js_eval import get_geolocation
from langchain_core.messages import AIMessage

# ==========================================
# 1. CONFIGURACIÓN Y CONEXIÓN
# ==========================================
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# ==========================================
# 2. MOTOR DE BÚSQUEDA
# ==========================================
def buscar_lugares_google(ciudad, palabra_clave, lat_gps=None, lon_gps=None):
    gmaps = obtener_cliente_google()
    
    if not gmaps: return None, "Error: No se detectó la API Key de Google."

    try:
        # A. CENTRO DEL MAPA
        if lat_gps is not None and lon_gps is not None:
            lat_centro, lon_centro = lat_gps, lon_gps
            origen_texto = "tu ubicación actual"
        else:
            geocode_result = gmaps.geocode(ciudad)
            if not geocode_result: return None, f"No encontré la ciudad: {ciudad}"
            loc = geocode_result[0]['geometry']['location']
            lat_centro, lon_centro = loc['lat'], loc['lng']
            origen_texto = ciudad

        # B. CONSULTAR LUGARES
        places_result = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000,
            open_now=False
        )

        lugares = places_result.get('results', [])
        if not lugares: return None, f"No encontré {palabra_clave} cerca de {origen_texto}."

        # C. CONSTRUIR DATOS
        data_completa = []

        # 1. TU UBICACIÓN (AZUL) 🔵
        if lat_gps is not None and lon_gps is not None:
            data_completa.append({
                'latitude': float(lat_gps),
                'longitude': float(lon_gps),
                'name': "🔵 TU UBICACIÓN",
                'address': "Estás aquí",
                'rating': "YO",
                'place_id': "USER_LOC",
                'color': '#0000FF',     # Azul Puro
                'size': 40              # Tamaño mediano
            })

        # 2. LUGARES (ROJO) 🔴
        for l in lugares:
            data_completa.append({
                'latitude': float(l['geometry']['location']['lat']),
                'longitude': float(l['geometry']['location']['lng']),
                'name': l.get('name', 'Sin nombre'),
                'address': l.get('vicinity', 'Dirección desconocida'),
                'rating': l.get('rating', 'N/A'),
                'place_id': l.get('place_id'),
                'color': '#FF0000', # Rojo Puro
                'size': 20          # Tamaño PEQUEÑO (Normal)
            })
            
        return pd.DataFrame(data_completa), f"✅ Encontré {len(lugares)} {palabra_clave}."

    except Exception as e:
        return None, f"Error: {str(e)}"

# ==========================================
# 3. INTERFAZ GRÁFICA
# ==========================================
def renderizar_sidebar():
    st.subheader("📍 Mapa Interactivo")
    
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None
    if "lugar_seleccionado_id" not in st.session_state:
        st.session_state.lugar_seleccionado_id = None

    # CONTROLES
    tipo = st.selectbox("Buscar:", ["Farmacias", "Hospitales", "Parques", "Gimnasios", "Restaurantes"])
    usar_gps = st.checkbox("📍 Usar mi GPS")
    
    lat_usuario, lon_usuario = None, None
    ciudad = ""

    if usar_gps:
        loc = get_geolocation()
        if loc:
            lat_usuario = loc['coords']['latitude']
            lon_usuario = loc['coords']['longitude']
            st.caption(f"📡 GPS OK")
        else:
            st.warning("⚠️ Esperando señal...")
    else:
        ciudad = st.text_input("Ciudad:", "Bogota")

    # BOTÓN BUSCAR
    if st.button("🔍 BUSCAR AHORA", use_container_width=True):
        if usar_gps and lat_usuario is None:
            st.error("GPS cargando...")
        else:
            with st.spinner("Buscando..."):
                df, msg = buscar_lugares_google(ciudad, tipo, lat_gps=lat_usuario, lon_gps=lon_usuario)
                if df is not None:
                    st.session_state.mapa_data = df
                    st.session_state.lugar_seleccionado_id = None
                    st.session_state.chat_history.append(AIMessage(content=f"[MAPA]: {msg}"))
                    st.rerun()
                else:
                    st.error(msg)

    # VISUALIZACIÓN
    if st.session_state.mapa_data is not None:
        
        st.divider()
        
        # 1. COPIA PARA VISUALIZAR
        df_display = st.session_state.mapa_data.copy()
        selected_id = st.session_state.lugar_seleccionado_id
        
        # 2. LOGICA DE RESALTADO (CORREGIDA)
        if selected_id:
            mask = df_display['place_id'] == selected_id
            df_display.loc[mask, 'color'] = '#00FF00' # Verde
            
            # --- AQUÍ ESTABA EL ERROR, LO HEMOS BAJADO A 80 ---
            df_display.loc[mask, 'size'] = 80         # Tamaño GRANDE (pero no gigante)
            
            st.info("🎯 Lugar resaltado en VERDE.")
            
            if st.button("🔙 Mapa normal", use_container_width=True):
                st.session_state.lugar_seleccionado_id = None
                st.rerun()

        # 3. PINTAR MAPA
        st.map(df_display, color='color', size='size', use_container_width=True)

        # 4. LISTA DE RESULTADOS
        st.caption("Resultados:")
        df_lista = st.session_state.mapa_data[st.session_state.mapa_data['place_id'] != "USER_LOC"].reset_index(drop=True)

        for i, row in df_lista.iterrows():
            with st.expander(f"🏥 {row['name']}"):
                st.write(f"📍 {row['address']}")
                
                # Botón SEÑALAR
                if st.button("🎯 SEÑALAR", key=f"btn_focus_{i}", use_container_width=True):
                    st.session_state.lugar_seleccionado_id = row['place_id']
                    st.rerun()

                # Botón GOOGLE MAPS
                link = f"https://www.google.com/maps/search/?api=1&query={row['name'].replace(' ', '+')}&query_place_id={row['place_id']}"
                st.link_button("🚗 IR (GOOGLE MAPS)", link, use_container_width=True)

        st.divider()
        if st.button("🗑️ Limpiar"):
            st.session_state.mapa_data = None
            st.session_state.lugar_seleccionado_id = None
            st.rerun()
