import streamlit as st
import pandas as pd
import googlemaps
import math
from streamlit_js_eval import get_geolocation
from langchain_core.messages import AIMessage

# ==========================================
# 1. MATEMÁTICAS: CALCULAR DISTANCIA (NUEVO)
# ==========================================
def calcular_distancia(lat1, lon1, lat2, lon2):
    # Fórmula de Haversine para distancia entre dos puntos GPS
    R = 6371  # Radio de la tierra en km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distancia = R * c
    return round(distancia, 2) # Devolvemos km con 2 decimales

# ==========================================
# 2. CONEXIÓN Y BÚSQUEDA
# ==========================================
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

def buscar_lugares_google(ciudad, palabra_clave, lat_gps=None, lon_gps=None):
    gmaps = obtener_cliente_google()
    
    if not gmaps: return None, "Error: No se detectó la API Key de Google."

    try:
        # A. DEFINIR PUNTO DE PARTIDA (REFERENCIA)
        if lat_gps is not None and lon_gps is not None:
            lat_ref, lon_ref = lat_gps, lon_gps
            origen_texto = "tu ubicación"
        else:
            geocode_result = gmaps.geocode(ciudad)
            if not geocode_result: return None, f"No encontré: {ciudad}"
            loc = geocode_result[0]['geometry']['location']
            lat_ref, lon_ref = loc['lat'], loc['lng']
            origen_texto = ciudad

        # B. CONSULTAR GOOGLE
        places_result = gmaps.places_nearby(
            location=(lat_ref, lon_ref),
            keyword=palabra_clave,
            radius=2000,
            open_now=False
        )

        lugares = places_result.get('results', [])
        if not lugares: return None, f"No encontré {palabra_clave}."

        # C. PROCESAR DATOS Y CALCULAR DISTANCIAS
        data_completa = []

        # 1. Agregamos al USUARIO (Distancia 0)
        if lat_gps is not None and lon_gps is not None:
            data_completa.append({
                'latitude': float(lat_ref),
                'longitude': float(lon_ref),
                'name': "🔵 TU UBICACIÓN",
                'address': "Estás aquí",
                'rating': "YO",
                'place_id': "USER_LOC",
                'color': '#0000FF',
                'size': 40,
                'distancia': 0.0 # Estás a 0 km de ti mismo
            })

        # 2. Agregamos los LUGARES y calculamos distancia
        for l in lugares:
            lat_lugar = l['geometry']['location']['lat']
            lon_lugar = l['geometry']['location']['lng']
            
            # --- CÁLCULO MÁGICO DE DISTANCIA ---
            km = calcular_distancia(lat_ref, lon_ref, lat_lugar, lon_lugar)

            data_completa.append({
                'latitude': float(lat_lugar),
                'longitude': float(lon_lugar),
                'name': l.get('name', 'Sin nombre'),
                'address': l.get('vicinity', ''),
                'rating': l.get('rating', 'N/A'),
                'place_id': l.get('place_id'),
                'color': '#FF0000',
                'size': 20,
                'distancia': km # <--- Guardamos la distancia
            })
            
        # D. CREAR DATAFRAME Y ORDENAR POR DISTANCIA
        df = pd.DataFrame(data_completa)
        # Ordenamos de menor a mayor distancia
        df = df.sort_values(by='distancia', ascending=True)
        
        return df, f"✅ Encontré {len(lugares)} sitios ordenados por cercanía."

    except Exception as e:
        return None, f"Error: {str(e)}"

# ==========================================
# 3. RENDERIZADO VISUAL
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
    
    if usar_gps:
        loc = get_geolocation()
        if loc:
            lat_usuario = loc['coords']['latitude']
            lon_usuario = loc['coords']['longitude']
            st.caption(f"📡 GPS OK")
        else:
            st.warning("⚠️ Esperando GPS...")
    else:
        st.text_input("Ciudad:", "Bogota") # Solo visual si no usa GPS

    # BOTÓN BUSCAR
    if st.button("🔍 BUSCAR CERCA", use_container_width=True):
        if usar_gps and lat_usuario is None:
            st.error("GPS cargando...")
        else:
            with st.spinner("Calculando distancias..."):
                df, msg = buscar_lugares_google("Bogota", tipo, lat_gps=lat_usuario, lon_gps=lon_usuario)
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
        
        # 1. VISUALIZAR MAPA
        df_display = st.session_state.mapa_data.copy()
        selected_id = st.session_state.lugar_seleccionado_id
        
        if selected_id:
            mask = df_display['place_id'] == selected_id
            df_display.loc[mask, 'color'] = '#00FF00'
            df_display.loc[mask, 'size'] = 80 
            st.info("🎯 Destino seleccionado.")
            
            if st.button("🔙 Mapa normal", use_container_width=True):
                st.session_state.lugar_seleccionado_id = None
                st.rerun()

        st.map(df_display, color='color', size='size', use_container_width=True)

        # 2. LISTA ORDENADA (LO MÁS IMPORTANTE)
        st.caption("Resultados (Ordenados por cercanía):")
        
        # Filtramos tu ubicación para que no salga en la lista
        df_lista = st.session_state.mapa_data[st.session_state.mapa_data['place_id'] != "USER_LOC"]
        # Aseguramos el orden (aunque ya viene ordenada, por seguridad)
        df_lista = df_lista.sort_values(by='distancia', ascending=True).reset_index(drop=True)

        for i, row in df_lista.iterrows():
            # --- AQUÍ MOSTRAMOS LA DISTANCIA EN EL TÍTULO ---
            distancia_str = f"{row['distancia']} km"
            
            # Usamos un ícono diferente para el más cercano (el primero de la lista)
            icono = "🥇" if i == 0 else "📍"
            
            with st.expander(f"{icono} {row['name']} ({distancia_str})"):
                st.write(f"🏠 {row['address']}")
                st.write(f"⭐ Calificación: {row['rating']}")
                
                if st.button("🎯 SEÑALAR", key=f"btn_focus_{i}", use_container_width=True):
                    st.session_state.lugar_seleccionado_id = row['place_id']
                    st.rerun()

                link = f"https://www.google.com/maps/search/?api=1&query={row['name'].replace(' ', '+')}&query_place_id={row['place_id']}"
                st.link_button("🚗 IR AHORA", link, use_container_width=True)

        st.divider()
        if st.button("🗑️ Limpiar"):
            st.session_state.mapa_data = None
            st.session_state.lugar_seleccionado_id = None
            st.rerun()
